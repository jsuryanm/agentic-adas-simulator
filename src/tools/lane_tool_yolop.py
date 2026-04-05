import os
import sys
import types
from typing import Optional, Tuple, List

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from src.core.config import settings
from src.models.schemas import LaneAnalysis
from src.exceptions.custom_exceptions import LaneToolException, ImageLoadException
from loguru import logger


class LaneToolYOLOP:
    """
    Detects lane lines using YOLOP's lane-line segmentation head.

    YOLOP produces three outputs in a single forward pass:
      1. Object detections  (unused here — handled by DetectionTool)
      2. Drivable-area mask
      3. Lane-line mask     ← this tool uses heads 2 & 3

    The lane mask is post-processed to extract left/right boundaries
    and compute lateral offset, matching the LaneAnalysis schema
    used by the rest of the pipeline.
    """

    YOLOP_INPUT_SIZE = (640, 640)

    # Normalisation constants (ImageNet, used by YOLOP)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self):
        self._model = None
        self._device = None

    # ── Model Loading ─────────────────────────────────────────────────

    @staticmethod
    def _patch_hub_imports() -> None:
        """
        YOLOP's torch-hub entry point imports its full training codebase,
        which drags in packages only needed for data loading / logging
        (prefetch_generator, tensorboard, …).  None of these are required
        for inference.

        We inject lightweight stubs into sys.modules *before*
        torch.hub.load triggers the import chain, so the real packages
        are never needed.
        """
        stubs = {
            "prefetch_generator": {"BackgroundGenerator": type("BackgroundGenerator", (), {
                "__init__": lambda self, gen, max_prefetch=1: setattr(self, "gen", gen),
                "__iter__": lambda self: iter(self.gen),
            })},
            "tensorboard": {},
            "tensorboard.SummaryWriter": {},
        }

        for mod_name, attrs in stubs.items():
            if mod_name not in sys.modules:
                dummy = types.ModuleType(mod_name)
                for attr_name, attr_val in attrs.items():
                    setattr(dummy, attr_name, attr_val)
                sys.modules[mod_name] = dummy

    def load_model(self) -> None:
        """Lazy-load YOLOP from torch hub on first use."""
        if self._model is not None:
            return

        try:
            self._patch_hub_imports()

            self._device = torch.device(
                settings.YOLO_DEVICE  # reuse the same device setting
            )
            self._model = torch.hub.load(
                "hustvl/yolop",
                "yolop",
                pretrained=True,
                trust_repo=True,
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"YOLOP model loaded on {self._device}")

        except Exception as e:
            raise LaneToolException(f"Failed to load YOLOP model: {e}") from e

    # ── Public API ────────────────────────────────────────────────────

    def detect_lanes(self, image_path: str) -> LaneAnalysis:
        """
        Detect lane lines and compute lateral offset using YOLOP.

        Args:
            image_path: Path to the driving frame.

        Returns:
            LaneAnalysis with lateral offset and lane geometry info.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ImageLoadException(f"Cannot read image: {image_path}")

        img_h, img_w = frame.shape[:2]
        self.load_model()

        try:
            # ── Inference ─────────────────────────────────────────
            input_tensor = self._preprocess(frame)

            with torch.no_grad():
                det_out, da_seg_out, ll_seg_out = self._model(input_tensor)

            # ── Lane-line mask ────────────────────────────────────
            lane_mask = self._decode_lane_mask(ll_seg_out, img_h, img_w)

            # ── Drivable-area mask (for road_coverage metric) ─────
            da_mask = self._decode_da_mask(da_seg_out, img_h, img_w)

            # ── Extract left / right lane x-coordinates ───────────
            left_xs, right_xs = self._extract_lane_boundaries(
                lane_mask, img_w, img_h
            )

            # ── Compute metrics ───────────────────────────────────
            lateral_offset = self._compute_lateral_offset(
                left_xs, right_xs, img_w, img_h
            )

            lane_width = self._compute_lane_width(left_xs, right_xs, img_h)

            lines_found = (1 if left_xs is not None else 0) + (
                1 if right_xs is not None else 0
            )

            road_coverage = self._compute_road_coverage(da_mask)

            analysis = LaneAnalysis(
                lateral_offset=round(max(-1.0, min(1.0, lateral_offset)), 4),
                lane_width_px=round(lane_width, 1) if lane_width else None,
                road_coverage=round(road_coverage, 4),
                road_detected=road_coverage > 0.02,
            )

            logger.info(
                f"Lane (YOLOP): offset={analysis.lateral_offset:.3f}, "
                f"lines_found={lines_found}/2, "
                f"road_coverage={analysis.road_coverage:.3f}"
            )
            return analysis

        except LaneToolException:
            raise
        except Exception as e:
            raise LaneToolException(f"YOLOP lane detection failed: {e}") from e

    # ── Preprocessing ─────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Resize, normalise, and batch the frame for YOLOP.

        YOLOP expects:
          - RGB float32 tensor
          - shape (1, 3, 640, 640)
          - ImageNet normalisation
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.YOLOP_INPUT_SIZE)

        # Normalise: (pixel / 255 - mean) / std
        img = resized.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD

        # HWC → CHW → NCHW
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self._device)

    # ── Mask Decoding ─────────────────────────────────────────────────

    def _decode_lane_mask(
        self,
        ll_seg_out: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> np.ndarray:
        """
        Decode YOLOP lane-line segmentation head into a binary mask
        at the original image resolution.
        """
        # ll_seg_out shape: (1, 2, H, W) — class 0=background, 1=lane
        probs = F.softmax(ll_seg_out, dim=1)
        lane_pred = probs[:, 1, :, :].squeeze(0).cpu().numpy()

        # Resize to original frame dimensions
        lane_resized = cv2.resize(
            lane_pred, (img_w, img_h), interpolation=cv2.INTER_LINEAR
        )

        # Binarise
        mask = (lane_resized > 0.5).astype(np.uint8)
        return mask

    def _decode_da_mask(
        self,
        da_seg_out: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> np.ndarray:
        """
        Decode YOLOP drivable-area segmentation head into a binary mask.
        """
        probs = F.softmax(da_seg_out, dim=1)
        da_pred = probs[:, 1, :, :].squeeze(0).cpu().numpy()

        da_resized = cv2.resize(
            da_pred, (img_w, img_h), interpolation=cv2.INTER_LINEAR
        )
        return (da_resized > 0.5).astype(np.uint8)

    # ── Lane Boundary Extraction ──────────────────────────────────────

    def _extract_lane_boundaries(
        self,
        lane_mask: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        From the binary lane mask, extract left and right lane-line
        x-coordinates for each row in the bottom half of the image.

        Returns:
            left_xs:  array of x-coords (one per row) or None
            right_xs: array of x-coords (one per row) or None

        Each array is indexed by row offset from the analysis start row.
        """
        img_cx = img_w // 2

        # Only analyse the bottom 50 % of the image (where lanes are visible)
        start_row = img_h // 2
        roi = lane_mask[start_row:, :]
        num_rows = roi.shape[0]

        left_xs = np.full(num_rows, np.nan, dtype=np.float32)
        right_xs = np.full(num_rows, np.nan, dtype=np.float32)

        for i in range(num_rows):
            row = roi[i, :]
            active = np.where(row > 0)[0]

            if len(active) == 0:
                continue

            # Split pixels into left-of-centre and right-of-centre
            left_pixels = active[active < img_cx]
            right_pixels = active[active >= img_cx]

            # Take the rightmost pixel on the left side (inner edge)
            if len(left_pixels) > 0:
                left_xs[i] = float(left_pixels[-1])

            # Take the leftmost pixel on the right side (inner edge)
            if len(right_pixels) > 0:
                right_xs[i] = float(right_pixels[0])

        # Require at least 15 % of rows to have valid data
        min_valid = int(num_rows * 0.15)

        left_valid = np.sum(~np.isnan(left_xs))
        right_valid = np.sum(~np.isnan(right_xs))

        left_out = left_xs if left_valid >= min_valid else None
        right_out = right_xs if right_valid >= min_valid else None

        return left_out, right_out

    # ── Metrics ───────────────────────────────────────────────────────

    def _compute_lateral_offset(
        self,
        left_xs: Optional[np.ndarray],
        right_xs: Optional[np.ndarray],
        img_w: int,
        img_h: int,
    ) -> float:
        """
        Compute normalised lateral offset from the ego-lane centre.

            0.0  = centred
           -1.0  = at the left lane boundary
           +1.0  = at the right lane boundary
        """
        img_cx = img_w / 2

        if left_xs is not None and right_xs is not None:
            # Use the bottom quarter for the most reliable estimate
            quarter = max(1, len(left_xs) // 4)
            left_bottom = float(np.nanmedian(left_xs[-quarter:]))
            right_bottom = float(np.nanmedian(right_xs[-quarter:]))

            lane_cx = (left_bottom + right_bottom) / 2
            lane_w = right_bottom - left_bottom

            if lane_w < 1:
                return 0.0

            return (img_cx - lane_cx) / (lane_w / 2)

        if left_xs is not None:
            quarter = max(1, len(left_xs) // 4)
            left_bottom = float(np.nanmedian(left_xs[-quarter:]))
            return -1.0 + 2.0 * (img_cx - left_bottom) / img_cx

        if right_xs is not None:
            quarter = max(1, len(right_xs) // 4)
            right_bottom = float(np.nanmedian(right_xs[-quarter:]))
            return (img_cx - right_bottom) / (img_w - img_cx)

        return 0.0

    def _compute_lane_width(
        self,
        left_xs: Optional[np.ndarray],
        right_xs: Optional[np.ndarray],
        img_h: int,
    ) -> Optional[float]:
        """
        Median lane width in pixels, measured in the bottom quarter
        of the analysis region.
        """
        if left_xs is None or right_xs is None:
            return None

        quarter = max(1, len(left_xs) // 4)
        left_slice = left_xs[-quarter:]
        right_slice = right_xs[-quarter:]

        # Only use rows where both sides have data
        valid = ~np.isnan(left_slice) & ~np.isnan(right_slice)
        if np.sum(valid) == 0:
            return None

        widths = right_slice[valid] - left_slice[valid]
        return float(np.median(widths))

    def _compute_road_coverage(self, da_mask: np.ndarray) -> float:
        """
        Fraction of the bottom half of the image covered by drivable area.
        Gives a richer road_coverage metric than the CV version's 0/0.5/1.
        """
        bottom_half = da_mask[da_mask.shape[0] // 2 :, :]
        if bottom_half.size == 0:
            return 0.0
        return float(np.sum(bottom_half) / bottom_half.size)


# ── Visual debugging ──────────────────────────────────────────────────────

if __name__ == "__main__":

    tool = LaneToolYOLOP()

    BASE_DIR = os.path.dirname(__file__)
    img_path = os.path.join(BASE_DIR, "images", "test_img2.jpg")

    # ── 1. Run the public API to get the LaneAnalysis ─────────────────
    analysis = tool.detect_lanes(img_path)

    print("\n===== LANE ANALYSIS (YOLOP) =====")
    print(f"  Lateral offset : {analysis.lateral_offset:.4f}")
    print(f"  Lane width (px): {analysis.lane_width_px}")
    print(f"  Road coverage  : {analysis.road_coverage:.4f}")
    print(f"  Road detected  : {analysis.road_detected}")

    # ── 2. Re-run inference to grab raw masks for visualisation ───────
    frame = cv2.imread(img_path)
    img_h, img_w = frame.shape[:2]

    input_tensor = tool._preprocess(frame)

    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = tool._model(input_tensor)

    lane_mask = tool._decode_lane_mask(ll_seg_out, img_h, img_w)
    da_mask = tool._decode_da_mask(da_seg_out, img_h, img_w)

    left_xs, right_xs = tool._extract_lane_boundaries(lane_mask, img_w, img_h)

    # ── 3. Build overlays ─────────────────────────────────────────────
    overlay = frame.copy()

    # -- Drivable area: green tint
    da_region = da_mask.astype(bool)
    if np.any(da_region):
        green_tint = overlay.copy()
        green_tint[:, :, 1] = np.clip(green_tint[:, :, 1].astype(int) + 120, 0, 255).astype(np.uint8)
        overlay[da_region] = cv2.addWeighted(
            frame, 0.7, green_tint, 0.3, 0
        )[da_region]

    # -- Lane lines: magenta tint
    lane_region = lane_mask.astype(bool)
    if np.any(lane_region):
        magenta = np.zeros_like(frame)
        magenta[:, :, 0] = 255  # blue
        magenta[:, :, 2] = 255  # red → magenta
        blended = cv2.addWeighted(frame, 0.4, magenta, 0.6, 0)
        overlay[lane_region] = blended[lane_region]

    # -- Draw extracted lane boundaries as polylines
    start_row = img_h // 2

    if left_xs is not None:
        pts = [
            (int(x), start_row + i)
            for i, x in enumerate(left_xs)
            if not np.isnan(x)
        ]
        if len(pts) > 1:
            cv2.polylines(
                overlay,
                [np.array(pts, dtype=np.int32)],
                isClosed=False,
                color=(255, 0, 0),  # blue
                thickness=2,
            )

    if right_xs is not None:
        pts = [
            (int(x), start_row + i)
            for i, x in enumerate(right_xs)
            if not np.isnan(x)
        ]
        if len(pts) > 1:
            cv2.polylines(
                overlay,
                [np.array(pts, dtype=np.int32)],
                isClosed=False,
                color=(0, 0, 255),  # red
                thickness=2,
            )

    # -- Image centre line (white dashed)
    for y in range(0, img_h, 20):
        cv2.line(
            overlay,
            (img_w // 2, y),
            (img_w // 2, min(y + 10, img_h)),
            (255, 255, 255),
            1,
        )

    # -- Lane centre line (yellow) if both boundaries exist
    if left_xs is not None and right_xs is not None:
        quarter = max(1, len(left_xs) // 4)
        left_bottom = float(np.nanmedian(left_xs[-quarter:]))
        right_bottom = float(np.nanmedian(right_xs[-quarter:]))
        lane_cx = int((left_bottom + right_bottom) / 2)
        cv2.line(overlay, (lane_cx, 0), (lane_cx, img_h), (0, 255, 255), 2)

    # -- HUD text
    lines_found = (1 if left_xs is not None else 0) + (1 if right_xs is not None else 0)
    offset_str = f"Offset: {analysis.lateral_offset:+.4f}"
    width_str = f"Lane width: {analysis.lane_width_px or 'N/A'}"
    road_str = f"Road coverage: {analysis.road_coverage:.2%}"
    lines_str = f"Lane lines: {lines_found}/2"

    y_text = 40
    for text in [offset_str, width_str, road_str, lines_str]:
        cv2.putText(
            overlay, text, (20, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        y_text += 35

    # ── 4. Build standalone mask windows ──────────────────────────────

    # Lane mask — magenta on black
    lane_vis = np.zeros_like(frame)
    lane_vis[lane_region, 0] = 255
    lane_vis[lane_region, 2] = 255

    # Drivable area mask — green on black
    da_vis = np.zeros_like(frame)
    da_vis[da_region, 1] = 200

    # Combined mask view
    combined_masks = cv2.addWeighted(da_vis, 0.6, lane_vis, 1.0, 0)

    # ── 5. Show windows ──────────────────────────────────────────────
    cv2.namedWindow("YOLOP Lane Overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Lane Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Drivable Area Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Combined Masks", cv2.WINDOW_NORMAL)

    cv2.imshow("YOLOP Lane Overlay", overlay)
    cv2.imshow("Lane Mask", lane_vis)
    cv2.imshow("Drivable Area Mask", da_vis)
    cv2.imshow("Combined Masks", combined_masks)

    cv2.waitKey(0)
    cv2.destroyAllWindows()