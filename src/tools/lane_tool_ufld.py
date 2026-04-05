# tools/lane_tool_ufld.py
# Lane detection using Ultra-Fast-Lane-Detection-v2 (UFLD-v2) via ONNX.
#
# Unlike SegFormer (detects road surface), UFLD detects lane *lines*,
# so it can distinguish the ego lane from adjacent lanes.
#
# Setup:
#   1. Download the ONNX model from PINTO model zoo (see README)
#   2. Place it at the path set in config.LANE_MODEL_PATH
#   3. Ensure onnxruntime is installed (already in your requirements.txt)
#
# How UFLD-v2 works (simplified):
#   The model divides each lane into 18 horizontal "slices" (row anchors).
#   For each slice, it predicts which column (out of 200 grid cells) the
#   lane passes through. We convert those grid positions to pixel (x, y)
#   coordinates → that gives us lane lines as point lists.

import os
from typing import Optional, Tuple, List

import numpy as np
import cv2

from src.core.config import settings
from src.models.schemas import LaneAnalysis
from src.exceptions.custom_exceptions import LaneToolException, ImageLoadException
from loguru import logger


# ── CULane ResNet-18 constants ────────────────────────────────────────────
# These are fixed for the ufldv2_culane_res18_320x1600.onnx model.
# If you use a different model variant, update these.

INPUT_H = 320        # model input height
INPUT_W = 1600       # model input width
NUM_LANES = 4        # max lanes the model detects
NUM_ROWS = 18        # number of horizontal slices per lane
NUM_COLS = 200       # number of grid columns per slice
ROW_ANCHORS = np.linspace(121, 301, NUM_ROWS).astype(int)  # y-positions in model space


class LaneToolUFLD:
    """Detects lane lines using UFLD-v2 and computes ego-lane lateral offset."""

    def __init__(self):
        self._session = None

    # ── Load model ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Lazy-load the ONNX model on first use."""
        if self._session is not None:
            return

        model_path = settings.LANE_MODEL_PATH
        if not os.path.isfile(model_path):
            raise LaneToolException(
                f"UFLD ONNX model not found at {model_path}. "
                "Download from PINTO model zoo."
            )

        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(model_path, providers=providers)
            logger.info(f"UFLD model loaded: {model_path}")
        except Exception as e:
            raise LaneToolException(f"Failed to load UFLD model: {e}") from e

    # ── Public API ────────────────────────────────────────────────────

    def detect_lanes(self, image_path: str) -> LaneAnalysis:
        """
        Detect lane lines and compute lateral offset.

        Returns LaneAnalysis with the same fields as SegFormer version,
        so SceneTool works with either approach.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ImageLoadException(f"Cannot read image: {image_path}")

        img_h, img_w = frame.shape[:2]
        self.load_model()

        try:
            # Step 1: Preprocess image for UFLD
            blob = self._preprocess(frame)

            # Step 2: Run ONNX inference
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: blob})

            # Step 3: Decode outputs into lane point lists
            lanes = self._decode_lanes(outputs, img_w, img_h)

            # Step 4: Find ego lane (the two lanes closest to image centre)
            left_lane, right_lane = self._find_ego_lane(lanes, img_w)

            # Step 5: Compute lateral offset
            lateral_offset = self._compute_offset(left_lane, right_lane, img_w, img_h)

            # Step 6: Lane width
            lane_width = None
            if left_lane is not None and right_lane is not None:
                left_x = self._lane_x_at_bottom(left_lane, img_h)
                right_x = self._lane_x_at_bottom(right_lane, img_h)
                if left_x is not None and right_x is not None:
                    lane_width = float(abs(right_x - left_x))

            analysis = LaneAnalysis(
                lateral_offset=round(max(-1.0, min(1.0, lateral_offset)), 4),
                lane_width_px=round(lane_width, 1) if lane_width else None,
                road_coverage=float(len(lanes) / NUM_LANES),  # 0 to 1
                road_detected=len(lanes) >= 2,
            )

            logger.info(
                f"UFLD: {len(lanes)} lanes, offset={analysis.lateral_offset:.3f}"
            )
            return analysis

        except LaneToolException:
            raise
        except Exception as e:
            raise LaneToolException(f"UFLD detection failed: {e}") from e

    # ── Step 1: Preprocess ────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalise (ImageNet), and reshape for ONNX."""
        resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        # HWC → NCHW
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        return blob.astype(np.float32)

    # ── Step 3: Decode ────────────────────────────────────────────────

    def _decode_lanes(
        self, outputs: list, img_w: int, img_h: int
    ) -> List[List[Tuple[float, float]]]:
        """
        Convert UFLD-v2 raw output into a list of lane lines.
        Each lane line = list of (x, y) pixel coordinates.

        UFLD-v2 outputs:
            outputs[0] = loc_row  (1, 4, 18, 200) — per-lane column probabilities
            outputs[1] = exist    (1, 4, 2)        — lane existence logits
        """
        loc_row = outputs[0]    # shape: (1, 4, 18, 200)
        exist = outputs[1]      # shape: (1, 4, 2)

        lanes = []

        for lane_id in range(NUM_LANES):
            # ── Check if this lane exists ──
            logits = exist[0, lane_id]
            logits = np.asarray(logits).reshape(-1)                     # shape: (2,)
            exp = np.exp(logits - np.max(logits))
            prob_exists = float(exp[1] / np.sum(exp))

            if prob_exists < 0.3:
                continue  # lane not detected

            # ── Decode this lane's points ──
            row_probs = loc_row[0, lane_id]                # shape: (18, 200)
            best_cols = np.argmax(row_probs, axis=1).astype(int)       # best column per row
            best_conf = np.max(row_probs, axis=1).astype(float)          # confidence per row

            points = []
            for row_idx in range(NUM_ROWS):
                if best_conf[row_idx] < 0.2:
                    continue  # low confidence, skip this row

                # Convert grid column → pixel x
                x = float(best_cols[row_idx]) / NUM_COLS * img_w

                # Convert row anchor → pixel y
                y = float(ROW_ANCHORS[row_idx]) / INPUT_H * img_h

                points.append((x, y))

            if len(points) >= 4:  # need at least 4 points for a lane
                lanes.append(points)

        return lanes

    # ── Step 4: Find ego lane ─────────────────────────────────────────

    def _find_ego_lane(
        self,
        lanes: List[List[Tuple[float, float]]],
        img_w: int,
    ) -> Tuple[Optional[List], Optional[List]]:
        """
        Find the two lanes forming the ego lane.

        Strategy: look at each lane's bottom-most point.
        - Left boundary = rightmost lane whose bottom point is left of centre
        - Right boundary = leftmost lane whose bottom point is right of centre
        """
        if len(lanes) < 2:
            return (lanes[0] if lanes else None, None)

        img_cx = img_w / 2
        left_candidates = []
        right_candidates = []

        for lane in lanes:
            bottom_x = max(lane, key=lambda p: p[1])[0]  # x of bottom-most point
            if bottom_x < img_cx:
                left_candidates.append(lane)
            else:
                right_candidates.append(lane)

        # Pick closest to centre on each side
        ego_left = None
        if left_candidates:
            ego_left = max(left_candidates, key=lambda ln: max(ln, key=lambda p: p[1])[0])

        ego_right = None
        if right_candidates:
            ego_right = min(right_candidates, key=lambda ln: max(ln, key=lambda p: p[1])[0])

        return ego_left, ego_right

    # ── Step 5: Lateral offset ────────────────────────────────────────

    def _lane_x_at_bottom(self, lane: List[Tuple[float, float]], img_h: int) -> Optional[float]:
        """Get the x-coordinate of a lane at the bottom of the image."""
        if not lane:
            return None
        # Use the point with the largest y (closest to bottom)
        return max(lane, key=lambda p: p[1])[0]

    def _compute_offset(
        self,
        left_lane: Optional[List],
        right_lane: Optional[List],
        img_w: int,
        img_h: int,
    ) -> float:
        """
        Same offset convention as SegFormer version:
            0.0  = centred in ego lane
           -1.0  = at left lane boundary
           +1.0  = at right lane boundary
        """
        img_cx = img_w / 2

        left_x = self._lane_x_at_bottom(left_lane, img_h) if left_lane else None
        right_x = self._lane_x_at_bottom(right_lane, img_h) if right_lane else None

        if left_x is not None and right_x is not None:
            lane_cx = (left_x + right_x) / 2
            lane_w = right_x - left_x
            if lane_w < 1:
                return 0.0
            return (img_cx - lane_cx) / (lane_w / 2)

        if left_x is not None:
            return -1.0 + 2.0 * (img_cx - left_x) / img_cx

        if right_x is not None:
            return (img_cx - right_x) / (img_w - img_cx)

        return 0.0

    # ── Public: get raw lane points (for visualization) ───────────────

    def get_lane_points(self, image_path: str) -> List[List[Tuple[float, float]]]:
        """Run detection and return raw lane point lists (for drawing)."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ImageLoadException(f"Cannot read image: {image_path}")

        img_h, img_w = frame.shape[:2]
        self.load_model()

        blob = self._preprocess(frame)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})

        return self._decode_lanes(outputs, img_w, img_h)


# ── Visual debugging ──────────────────────────────────────────────────────

if __name__ == "__main__":

    tool = LaneToolUFLD()

    BASE_DIR = os.path.dirname(__file__)
    img_path = os.path.join(BASE_DIR, "images", "test.jpg")

    # Run analysis
    analysis = tool.detect_lanes(img_path)
    print("\n===== LANE ANALYSIS (UFLD) =====")
    print("Lateral offset:", analysis.lateral_offset)
    print("Lane width:", analysis.lane_width_px)
    print("Lanes detected:", analysis.road_coverage)
    print("Road detected:", analysis.road_detected)

    # Get raw lane points for drawing
    lanes = tool.get_lane_points(img_path)
    frame = cv2.imread(img_path)
    img_h, img_w = frame.shape[:2]

    # Draw each lane in a different colour
    colours = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]

    for i, lane in enumerate(lanes):
        colour = colours[i % len(colours)]

        # Draw points
        for (x, y) in lane:
            cv2.circle(frame, (int(x), int(y)), 5, colour, -1)

        # Connect points with lines
        for j in range(len(lane) - 1):
            pt1 = (int(lane[j][0]), int(lane[j][1]))
            pt2 = (int(lane[j + 1][0]), int(lane[j + 1][1]))
            cv2.line(frame, pt1, pt2, colour, 2)

    # Draw image centre
    cv2.line(frame, (img_w // 2, 0), (img_w // 2, img_h), (255, 255, 255), 2)

    cv2.putText(
        frame,
        f"Offset: {analysis.lateral_offset:.4f}  Lanes: {len(lanes)}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
    )

    cv2.imshow("UFLD Lane Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()