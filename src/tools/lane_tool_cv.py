# tools/lane_tool_cv.py
# Classical lane detection using OpenCV (Canny + Hough Lines).
#
# No model download, no ONNX, no transformers — pure computer vision.
# Detects lane *lines* (not road surface), so it can distinguish
# the ego lane from adjacent lanes, solving the multi-lane problem
# that SegFormer has.
#
# Pipeline:
#   image → grayscale → blur → Canny edges → ROI mask
#   → Hough lines → separate left/right → fit lines → lateral offset

import os
from typing import Optional, Tuple, List

import numpy as np
import cv2

from src.core.config import settings
from src.models.schemas import LaneAnalysis
from src.exceptions.custom_exceptions import LaneToolException, ImageLoadException
from loguru import logger


class LaneToolCV:
    """
    Detects lane lines using classical CV and computes lateral offset.

    Unlike the SegFormer approach, this detects lane *markings*,
    so it works per-lane rather than per-road.
    """

    def __init__(self):
        # Canny thresholds
        self.canny_low = 50
        self.canny_high = 150

        # Hough line parameters
        self.hough_threshold = 30
        self.min_line_length = 40
        self.max_line_gap = 150

        # Slope filters (reject near-horizontal lines)
        self.min_slope = 0.4
        self.max_slope = 3.0

    def detect_lanes(self, image_path: str) -> LaneAnalysis:
        """
        Detect lane lines and compute lateral offset.

        Args:
            image_path: Path to the driving frame.

        Returns:
            LaneAnalysis with lateral offset and lane info.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ImageLoadException(f"Cannot read image: {image_path}")

        img_h, img_w = frame.shape[:2]

        try:
            # 1. Detect edges
            edges = self._detect_edges(frame)

            # 2. Mask to region of interest (bottom triangle)
            masked = self._apply_roi(edges, img_h, img_w)

            # 3. Find lines with Hough transform
            lines = cv2.HoughLinesP(
                masked,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap,
            )

            # 4. Separate into left and right lane lines
            left_lines, right_lines = self._separate_lines(lines, img_w)

            # 5. Fit a single line for each side
            left_line = self._fit_lane_line(left_lines, img_h)
            right_line = self._fit_lane_line(right_lines, img_h)

            # 6. Compute lateral offset
            lateral_offset = self._compute_lateral_offset(
                left_line, right_line, img_w, img_h
            )

            # 7. Lane width at bottom of image
            lane_width = None
            if left_line is not None and right_line is not None:
                left_x_bottom = left_line[0]
                right_x_bottom = right_line[0]
                lane_width = float(abs(right_x_bottom - left_x_bottom))

            # 8. Confidence based on how many lines we found
            lines_found = (1 if left_line else 0) + (1 if right_line else 0)
            road_detected = lines_found > 0

            analysis = LaneAnalysis(
                lateral_offset=round(max(-1.0, min(1.0, lateral_offset)), 4),
                lane_width_px=round(lane_width, 1) if lane_width else None,
                road_coverage=float(lines_found / 2),  # 0, 0.5, or 1.0
                road_detected=road_detected,
            )

            logger.info(
                f"Lane (CV): offset={analysis.lateral_offset:.3f}, "
                f"lines_found={lines_found}/2"
            )
            return analysis

        except LaneToolException:
            raise
        except Exception as e:
            raise LaneToolException(f"Lane detection failed: {e}") from e

    # ── Edge Detection ────────────────────────────────────────────────────

    def _detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale, blur, and run Canny edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges

    # ── Region of Interest ────────────────────────────────────────────────

    def _apply_roi(self, edges: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
        """
        Mask edges to only keep the bottom trapezoidal region
        where lane lines are expected.

        Shape:
            Top edge:    narrow band around the centre (horizon area)
            Bottom edge: full frame width
        """
        mask = np.zeros_like(edges)

        # Trapezoid vertices
        roi = np.array([[
            (0, img_h),                          # bottom-left
            (int(img_w * 0.4), int(img_h * 0.55)),  # top-left
            (int(img_w * 0.6), int(img_h * 0.55)),  # top-right
            (img_w, img_h),                      # bottom-right
        ]], dtype=np.int32)

        cv2.fillPoly(mask, roi, 255)
        return cv2.bitwise_and(edges, mask)

    # ── Line Separation ───────────────────────────────────────────────────

    def _separate_lines(
        self, lines: Optional[np.ndarray], img_w: int
    ) -> Tuple[List, List]:
        """
        Split detected lines into left-lane and right-lane groups
        based on their slope and position.

        Left lane lines:  negative slope (goes up-right in image coords)
        Right lane lines: positive slope (goes up-left in image coords)
        """
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        img_cx = img_w / 2

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Avoid division by zero
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter out near-horizontal lines (not lane markings)
            if abs(slope) < self.min_slope or abs(slope) > self.max_slope:
                continue

            # Midpoint of the line segment
            mid_x = (x1 + x2) / 2

            # Left lane: negative slope AND on the left half
            if slope < 0 and mid_x < img_cx:
                left_lines.append((x1, y1, x2, y2))

            # Right lane: positive slope AND on the right half
            elif slope > 0 and mid_x > img_cx:
                right_lines.append((x1, y1, x2, y2))

        return left_lines, right_lines

    # ── Line Fitting ──────────────────────────────────────────────────────

    def _fit_lane_line(
        self, lines: List[Tuple], img_h: int
    ) -> Optional[Tuple[float, float]]:
        """
        Fit all line segments on one side into a single averaged line.

        Uses np.polyfit (degree 1) on all the endpoint coordinates.

        Returns:
            (x_bottom, x_top) — the x-coordinates of the fitted line
            at the bottom and top of the ROI, or None if no lines found.
        """
        if not lines:
            return None

        # Collect all points from all line segments
        all_x = []
        all_y = []
        for x1, y1, x2, y2 in lines:
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        if len(all_x) < 2:
            return None

        try:
            # Fit: x = f(y) — we fit x as function of y because
            # lane lines are more vertical than horizontal
            coeffs = np.polyfit(all_y, all_x, deg=1)
            poly = np.poly1d(coeffs)

            y_bottom = img_h
            y_top = int(img_h * 0.55)  # matches ROI top

            x_bottom = float(poly(y_bottom))
            x_top = float(poly(y_top))

            return (x_bottom, x_top)

        except (np.linalg.LinAlgError, ValueError):
            return None

    # ── Lateral Offset ────────────────────────────────────────────────────

    def _compute_lateral_offset(
        self,
        left_line: Optional[Tuple[float, float]],
        right_line: Optional[Tuple[float, float]],
        img_w: int,
        img_h: int,
    ) -> float:
        """
        Compute lateral offset from the ego lane centre.

        Same convention as SegFormer version:
            0.0  = centred
           -1.0  = at the left lane boundary
           +1.0  = at the right lane boundary
        """
        img_cx = img_w / 2

        if left_line is not None and right_line is not None:
            # Use x-coordinates at the bottom of the image
            left_x = left_line[0]
            right_x = right_line[0]
            lane_cx = (left_x + right_x) / 2
            lane_w = right_x - left_x

            if lane_w < 1:
                return 0.0

            return (img_cx - lane_cx) / (lane_w / 2)

        # Only one line detected — rough estimate from that side
        if left_line is not None:
            left_x = left_line[0]
            return -1.0 + 2.0 * (img_cx - left_x) / img_cx

        if right_line is not None:
            right_x = right_line[0]
            return (img_cx - right_x) / (img_w - img_cx)

        return 0.0


# ── Visual debugging ──────────────────────────────────────────────────────

if __name__ == "__main__":

    tool = LaneToolCV()

    BASE_DIR = os.path.dirname(__file__)
    img_path = os.path.join(BASE_DIR, "images", "test.jpg")

    # Run analysis
    analysis = tool.detect_lanes(img_path)
    print("\n===== LANE ANALYSIS (OpenCV) =====")
    print("Lateral offset:", analysis.lateral_offset)
    print("Lane width:", analysis.lane_width_px)
    print("Road coverage:", analysis.road_coverage)
    print("Road detected:", analysis.road_detected)

    # ── Visualization ─────────────────────────────────────────────────
    frame = cv2.imread(img_path)
    img_h, img_w = frame.shape[:2]
    overlay = frame.copy()

    # Show edges + ROI
    edges = tool._detect_edges(frame)
    masked = tool._apply_roi(edges, img_h, img_w)

    # Get lines and fit
    lines = cv2.HoughLinesP(
        masked, 1, np.pi / 180,
        tool.hough_threshold,
        minLineLength=tool.min_line_length,
        maxLineGap=tool.max_line_gap,
    )
    left_lines, right_lines = tool._separate_lines(lines, img_w)
    left_fit = tool._fit_lane_line(left_lines, img_h)
    right_fit = tool._fit_lane_line(right_lines, img_h)

    # Draw raw Hough lines (thin, gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Draw fitted left lane (blue)
    if left_fit is not None:
        x_bot, x_top = left_fit
        y_bot, y_top = img_h, int(img_h * 0.55)
        cv2.line(overlay, (int(x_bot), y_bot), (int(x_top), y_top), (255, 0, 0), 3)

    # Draw fitted right lane (red)
    if right_fit is not None:
        x_bot, x_top = right_fit
        y_bot, y_top = img_h, int(img_h * 0.55)
        cv2.line(overlay, (int(x_bot), y_bot), (int(x_top), y_top), (0, 0, 255), 3)

    # Draw image centre (white)
    cv2.line(overlay, (img_w // 2, 0), (img_w // 2, img_h), (255, 255, 255), 2)

    # Draw lane centre (yellow)
    if left_fit and right_fit:
        lane_cx = int((left_fit[0] + right_fit[0]) / 2)
        cv2.line(overlay, (lane_cx, 0), (lane_cx, img_h), (0, 255, 255), 2)

    # Draw ROI trapezoid
    roi_pts = np.array([
        (0, img_h),
        (int(img_w * 0.4), int(img_h * 0.55)),
        (int(img_w * 0.6), int(img_h * 0.55)),
        (img_w, img_h),
    ], dtype=np.int32)
    cv2.polylines(overlay, [roi_pts], True, (0, 255, 0), 2)

    cv2.putText(
        overlay,
        f"Offset: {analysis.lateral_offset:.4f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
    )

    # Show edges
    edges_color = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Edges (ROI masked)", edges_color)
    cv2.imshow("Lane Analysis (OpenCV)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()