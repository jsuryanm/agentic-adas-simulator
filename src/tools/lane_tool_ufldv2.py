import os
from typing import List, Optional, Tuple

import numpy as np
import cv2

from src.core.config import settings
from src.exceptions.custom_exceptions import ImageLoadException, LaneToolException
from loguru import logger
from src.models.schemas import LaneAnalysis, LanePoint, LaneLine


class LaneTool:
    """
    Detects lane lines and computes ego-lane lateral offset.

    Pipeline:
        image → preprocess → ONNX inference → softmax → decode row anchors
        → filter spurious lanes → identify ego lane → lateral offset

    This tool wraps the UFLDv2 (CULane ResNet-18 320×1600) ONNX model.
    The model outputs row-anchor predictions: for each of 4 lane slots
    and N horizontal slices, it predicts which grid column the lane passes
    through.  We convert those to pixel coordinates in the original frame.
    """

    def __init__(self):
        self._session = None
        self.input_h: int = settings.LANE_INPUT_HEIGHT   # 320
        self.input_w: int = settings.LANE_INPUT_WIDTH    # 1600

    # ── Model lifecycle ───────────────────────────────────────────────────

    def load_model(self) -> None:
        """Lazy-load the ONNX model on first use."""
        if self._session is not None:
            return

        model_path = settings.LANE_MODEL_PATH
        if not os.path.isfile(model_path):
            raise LaneToolException(
                f"Lane ONNX model not found at {model_path}. "
                "Download or export from the UFLD-v2 repo."
            )

        logger.info(f"Loading lane detection model: {model_path}")
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(model_path, providers=providers)
            logger.info("Lane model loaded successfully")

        except Exception as e:
            raise LaneToolException(f"Failed to load lane model: {e}") from e

    # ── Public API ────────────────────────────────────────────────────────

    def detect_lanes(self, image_path: str) -> LaneAnalysis:
        """
        Detect lane lines in a driving frame and compute lateral offset.

        Args:
            image_path: Path to the image file.

        Returns:
            LaneAnalysis with detected lanes, ego-lane IDs, and offset.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ImageLoadException(f"Cannot read image: {image_path}")

        img_h, img_w = frame.shape[:2]
        self.load_model()

        try:
            blob = self._preprocess(frame)
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: blob})

            # ── Interpret raw outputs ─────────────────────────────────
            loc_row = outputs[0]
            # outputs[1] is loc_col (column-anchor preds), NOT existence.
            # Shape check: existence would be (1, 2, N, 4); loc_col is
            # typically (1, num_gridding_col, num_cls_col, num_lanes).
            # We only need loc_row for lane detection.

            # Fix ONNX axis order: exported models sometimes have the
            # gridding dimension in axis-1 instead of axis-3.
            # Target shape: (1, num_lanes, num_rows, num_gridding)
            if loc_row.ndim == 4 and loc_row.shape[3] <= 8:
                # Shape is (1, gridding, rows, lanes) → transpose
                loc_row = np.transpose(loc_row, (0, 3, 2, 1))

            num_lanes = loc_row.shape[1]
            num_rows = loc_row.shape[2]
            num_gridding = loc_row.shape[3]

            logger.debug(
                f"UFLD output: lanes={num_lanes}, "
                f"rows={num_rows}, gridding={num_gridding}"
            )

            # Build row anchor fractions for this model variant.
            # CULane anchors span from ~42 % to ~99 % of the input height.
            row_anchor_fracs = np.linspace(
                settings.LANE_ROW_ANCHOR_START,  # 0.42
                settings.LANE_ROW_ANCHOR_END,    # 0.99
                num_rows,
            )

            # ── Decode lanes ──────────────────────────────────────────
            lanes = self._decode_lanes(
                loc_row, num_lanes, num_rows, num_gridding,
                row_anchor_fracs, img_w, img_h,
            )

            # ── Ego-lane & offset ─────────────────────────────────────
            ego_left, ego_right = self._identify_ego_lane(lanes, img_w)

            lateral_offset = self._compute_lateral_offset(
                ego_left, ego_right, img_w,
            )

            lane_width = None
            if ego_left is not None and ego_right is not None:
                lane_width = self._estimate_lane_width(
                    ego_left, ego_right, img_h,
                )

            analysis = LaneAnalysis(
                lanes=lanes,
                ego_left_lane_id=(
                    ego_left.lane_id if ego_left is not None else None
                ),
                ego_right_lane_id=(
                    ego_right.lane_id if ego_right is not None else None
                ),
                lateral_offset=round(
                    max(-1.0, min(1.0, lateral_offset)), 4
                ),
                lane_width_px=(
                    round(lane_width, 1) if lane_width is not None else None
                ),
            )

            logger.info(
                f"Detected {len(lanes)} lane(s), "
                f"lateral_offset={analysis.lateral_offset:.3f}"
            )
            return analysis

        except LaneToolException:
            raise
        except Exception as e:
            raise LaneToolException(f"Lane detection failed: {e}") from e

    # ── Preprocessing ─────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalise (ImageNet), and batch the image for UFLD-v2."""
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb / 255.0 - mean) / std

        # HWC → CHW → NCHW
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        return blob.astype(np.float32)

    # ── Decoding ──────────────────────────────────────────────────────────

    def _decode_lanes(
        self,
        loc_row: np.ndarray,
        num_lanes: int,
        num_rows: int,
        num_gridding: int,
        row_anchor_fracs: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> List[LaneLine]:
        """
        Decode UFLDv2 loc_row logits into lane lines.

        For each lane slot:
          1. Apply softmax across grid columns to get real probabilities.
          2. Take argmax as the predicted column, max prob as confidence.
          3. Filter low-confidence rows.
          4. Convert (grid_col, row_anchor) → (pixel_x, pixel_y).
          5. Reject lanes with too few points or non-monotonic x.

        Args:
            loc_row:          (1, num_lanes, num_rows, num_gridding) logits.
            num_lanes:        Number of lane slots (typically 4).
            num_rows:         Number of row anchors in this model variant.
            num_gridding:     Number of grid columns per row anchor.
            row_anchor_fracs: Row anchor positions as fractions of image height.
            img_w, img_h:     Original image dimensions.

        Returns:
            List of LaneLine objects, sorted left-to-right.
        """
        candidates: List[Tuple[float, List[LanePoint], float]] = []

        for lane_id in range(num_lanes):
            logits = loc_row[0, lane_id]  # (num_rows, num_gridding)

            # ── BUG FIX 1: apply softmax to convert logits → probabilities
            probs = self._softmax(logits, axis=1)

            col_indices = np.argmax(probs, axis=1)
            col_confidence = np.max(probs, axis=1)

            # ── BUG FIX 2: lane existence proxy ──────────────────────
            # No existence tensor available in this ONNX export.
            # Use mean confidence of the top-K rows as a proxy.
            top_k = min(10, num_rows)
            sorted_conf = np.sort(col_confidence)[::-1]
            lane_confidence = float(np.mean(sorted_conf[:top_k]))

            if lane_confidence < settings.LANE_EXISTENCE_THRESHOLD:
                logger.debug(
                    f"Lane {lane_id} rejected: confidence={lane_confidence:.4f} "
                    f"< threshold={settings.LANE_EXISTENCE_THRESHOLD}"
                )
                continue  # this lane slot is probably empty

            logger.debug(
                f"Lane {lane_id} accepted: confidence={lane_confidence:.4f}, "
                f"max_row_conf={float(sorted_conf[0]):.4f}"
            )

            # ── Decode points ─────────────────────────────────────────
            xs: List[float] = []
            ys: List[float] = []
            confs: List[float] = []

            for row_idx in range(num_rows):
                if col_confidence[row_idx] < settings.LANE_POINT_THRESHOLD:
                    continue

                # BUG FIX 3: use +0.5 for grid-cell centre, and use
                # fractional anchors mapped directly to image height
                x = (float(col_indices[row_idx]) + 0.5) / num_gridding * img_w
                y = float(row_anchor_fracs[row_idx]) * img_h

                xs.append(x)
                ys.append(y)
                confs.append(float(col_confidence[row_idx]))

            if len(xs) < settings.LANE_MIN_POINTS:
                logger.debug(
                    f"Lane {lane_id}: only {len(xs)} points "
                    f"(need {settings.LANE_MIN_POINTS}), skipped"
                )
                continue

            # Sort by y ascending (top of image → bottom)
            order = np.argsort(ys)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            confs = np.array(confs)[order]

            # ── BUG FIX 4: monotonicity check ────────────────────────
            # A valid lane's x-coordinates should be roughly monotonic
            # (either consistently increasing or decreasing) as y grows.
            # Wild zigzags indicate noise.  We allow a small number of
            # direction changes.
            if not self._is_roughly_monotonic(xs, max_reversals=3):
                logger.debug(f"Lane {lane_id}: failed monotonicity check, skipped")
                continue

            # ── BUG FIX 5: span check ────────────────────────────────
            # Lane must span a meaningful vertical range
            y_span = ys[-1] - ys[0]
            if y_span < img_h * 0.10:
                logger.debug(
                    f"Lane {lane_id}: y_span={y_span:.0f}px "
                    f"< {img_h * 0.10:.0f}px, skipped"
                )
                continue

            # Build LanePoint list (NO polynomial smoothing — the raw
            # softmax-decoded points are accurate enough, and polyfit
            # on noisy data was causing the loops/crossings).
            bottom_x = float(xs[-1])
            avg_conf = float(np.mean(confs))

            points = [
                LanePoint(x=round(float(xs[i]), 1),
                          y=round(float(ys[i]), 1))
                for i in range(len(xs))
            ]

            candidates.append((bottom_x, points, avg_conf))

        # Sort lanes left → right by their bottom-most x
        candidates.sort(key=lambda c: c[0])

        lanes = [
            LaneLine(
                lane_id=i,
                points=pts,
                confidence=round(conf, 3),
            )
            for i, (_, pts, conf) in enumerate(candidates)
        ]

        return lanes

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
        """Numerically stable softmax along the given axis."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @staticmethod
    def _is_roughly_monotonic(xs: np.ndarray, max_reversals: int = 3) -> bool:
        """
        Check whether a sequence of x-values is roughly monotonic.

        Allows up to `max_reversals` direction changes to tolerate
        minor noise, but rejects wildly zigzagging sequences.
        """
        if len(xs) < 3:
            return True

        diffs = np.diff(xs)
        # Ignore near-zero diffs (< 2 px jitter)
        significant = diffs[np.abs(diffs) > 2.0]

        if len(significant) < 2:
            return True

        signs = np.sign(significant)
        reversals = int(np.sum(np.abs(np.diff(signs)) > 0))

        return reversals <= max_reversals

    # ── Ego-lane identification ───────────────────────────────────────────

    def _identify_ego_lane(
        self,
        lanes: List[LaneLine],
        img_w: int,
    ) -> Tuple[Optional[LaneLine], Optional[LaneLine]]:
        """
        Find the two lanes forming the ego lane (the lane the vehicle
        is currently driving in).

        Strategy: for each lane, look at its bottom-most point.
        - Left boundary  = rightmost lane whose bottom point is left of centre.
        - Right boundary = leftmost lane whose bottom point is right of centre.
        """
        if len(lanes) < 2:
            return (lanes[0] if lanes else None, None)

        center = img_w / 2
        left: Optional[LaneLine] = None
        right: Optional[LaneLine] = None
        left_dist = float("inf")
        right_dist = float("inf")

        for lane in lanes:
            bottom_x = max(lane.points, key=lambda p: p.y).x

            if bottom_x < center:
                dist = center - bottom_x
                if dist < left_dist:
                    left_dist = dist
                    left = lane
            else:
                dist = bottom_x - center
                if dist < right_dist:
                    right_dist = dist
                    right = lane

        return left, right

    # ── Lateral offset ────────────────────────────────────────────────────

    def _compute_lateral_offset(
        self,
        left_lane: Optional[LaneLine],
        right_lane: Optional[LaneLine],
        img_w: int,
    ) -> float:
        """
        Compute normalised lateral offset from ego-lane centre.

        Returns:
            float in [-1, 1].
             0.0  = perfectly centred
            -1.0  = touching the left boundary
            +1.0  = touching the right boundary
        """
        img_cx = img_w / 2

        if left_lane is not None and right_lane is not None:
            left_x = max(left_lane.points, key=lambda p: p.y).x
            right_x = max(right_lane.points, key=lambda p: p.y).x
            lane_cx = (left_x + right_x) / 2
            lane_w = right_x - left_x

            if lane_w < 1:
                return 0.0
            return (img_cx - lane_cx) / (lane_w / 2)

        if left_lane is not None:
            left_x = max(left_lane.points, key=lambda p: p.y).x
            return -1.0 + 2.0 * (img_cx - left_x) / img_cx

        if right_lane is not None:
            right_x = max(right_lane.points, key=lambda p: p.y).x
            return (img_cx - right_x) / (img_w - img_cx)

        return 0.0

    # ── Lane width ────────────────────────────────────────────────────────

    def _estimate_lane_width(
        self,
        left_lane: LaneLine,
        right_lane: LaneLine,
        img_h: int,
    ) -> float:
        """Estimate ego-lane width using points in the bottom quarter."""
        threshold_y = img_h * 0.75

        left_bottom = [p for p in left_lane.points if p.y >= threshold_y]
        right_bottom = [p for p in right_lane.points if p.y >= threshold_y]

        if not left_bottom or not right_bottom:
            left_x = np.mean([p.x for p in left_lane.points])
            right_x = np.mean([p.x for p in right_lane.points])
        else:
            left_x = np.mean([p.x for p in left_bottom])
            right_x = np.mean([p.x for p in right_bottom])

        return float(abs(right_x - left_x))


# ── Visual debugging ──────────────────────────────────────────────────────

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(__file__)
    image_path = os.path.join(BASE_DIR, "images", "test.jpg")

    try:
        lane_tool = LaneTool()
        analysis = lane_tool.detect_lanes(image_path)

        frame = cv2.imread(image_path)
        if frame is None:
            raise Exception("Could not load test image")

        # Draw all detected lanes
        for lane in analysis.lanes:
            is_ego = (
                lane.lane_id == analysis.ego_left_lane_id
                or lane.lane_id == analysis.ego_right_lane_id
            )
            color = (0, 255, 0) if is_ego else (0, 0, 255)
            thickness = 5 if is_ego else 3

            for pt in lane.points:
                cv2.circle(frame, (int(pt.x), int(pt.y)), 5, color, -1)

            pts = np.array([(int(p.x), int(p.y)) for p in lane.points])
            cv2.polylines(frame, [pts], False, color, thickness)

        # Draw vehicle centre line
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, h), (w // 2, h - 150), (255, 0, 0), 3)

        cv2.putText(
            frame,
            f"Offset: {analysis.lateral_offset}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Lane Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("\n===== LANE ANALYSIS =====")
        print(f"Detected lanes: {len(analysis.lanes)}")
        print(f"Ego left lane:  {analysis.ego_left_lane_id}")
        print(f"Ego right lane: {analysis.ego_right_lane_id}")
        print(f"Lateral offset: {analysis.lateral_offset}")
        print(f"Lane width:     {analysis.lane_width_px}")

    except Exception as e:
        print(f"Error: {e}")