# tools/lane_tool_advanced.py
# Advanced lane detection using classical CV with bird's-eye perspective.
#
# Adapted from the Udacity-style "Advanced Lane Finding" pipeline,
# re-engineered for arbitrary dashcam footage (no camera calibration needed).
#
# Pipeline:
#   image → HLS S-channel + Sobel-X thresholding → combined binary
#   → adaptive perspective warp (bird's-eye) → sliding-window histogram
#   → 2nd-degree polynomial fit → curvature + lateral offset
#
# Key differences from the notebook version:
#   - No camera calibration / undistortion (requires chessboard images)
#   - Perspective transform points are computed dynamically from image dims
#   - Sliding window params are configurable via Config
#   - Returns LaneAnalysis (same schema as SegFormer / CV / UFLD tools)
#   - Adds curvature_radius for RiskAgent consumption

import os
from typing import Optional, Tuple, List

import numpy as np
import cv2

from src.core.config import settings
from src.models.schemas import LaneAnalysis
from src.exceptions.custom_exceptions import LaneToolException, ImageLoadException
from loguru import logger


class LaneToolAdvanced:
    """
    Detects lane lines using HLS+Sobel thresholding, bird's-eye perspective
    transform, and sliding-window polynomial fitting.

    Outputs curvature radius alongside lateral offset, giving the RiskAgent
    an additional signal for curve-ahead warnings.
    """

    def __init__(self):
        # ── Thresholding ──────────────────────────────────────────────
        self.s_thresh = (90, 255)       # HLS S-channel range
        self.sx_thresh = (30, 100)      # Sobel-X gradient range
        self.sobel_kernel = 3

        # ── Perspective transform (proportional to image dims) ────────
        # Source trapezoid: the region on the road in the original image.
        # Expressed as (x_ratio, y_ratio) of image width/height.
        # These proportions work well for typical forward-facing dashcams.
        self.src_ratios = [
            (0.18, 0.94),   # bottom-left
            (0.43, 0.65),   # top-left
            (0.57, 0.65),   # top-right
            (0.85, 0.94),   # bottom-right
        ]
        # Destination rectangle in the warped image
        self.dst_x_margin = 0.25   # 25% margin on each side
        self.dst_y_top = 0.0
        self.dst_y_bottom = 1.0

        # ── Sliding window ────────────────────────────────────────────
        self.nwindows = settings.LANE_ADV_NWINDOWS
        self.margin = settings.LANE_ADV_MARGIN
        self.minpix = settings.LANE_ADV_MINPIX

        # ── Real-world conversion (approximate) ──────────────────────
        # US highway lane ≈ 3.7 m wide, visible stretch ≈ 30 m
        self.ym_per_pix = 30.0 / 720
        self.xm_per_pix = 3.7 / 700

    # ── Public API ────────────────────────────────────────────────────

    def detect_lanes(self, image_path: str) -> LaneAnalysis:
        """
        Full lane detection pipeline on a single frame.

        Args:
            image_path: Path to the driving frame.

        Returns:
            LaneAnalysis with lateral offset, lane width, curvature, etc.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ImageLoadException(f"Cannot read image: {image_path}")

        img_h, img_w = frame.shape[:2]

        try:
            # 1. Create combined binary (HLS S-channel + Sobel-X)
            binary = self._threshold_binary(frame)

            # 2. Perspective warp to bird's-eye view
            binary_warped, M, Minv = self._perspective_warp(binary, img_w, img_h)

            # 3. Sliding window search for lane pixels
            left_x, left_y, right_x, right_y = self._sliding_window_search(
                binary_warped
            )

            # 4. Fit 2nd-degree polynomials
            left_fit, right_fit = self._fit_polynomials(
                left_x, left_y, right_x, right_y
            )

            # 5. Compute lateral offset
            lateral_offset = self._compute_lateral_offset(
                left_fit, right_fit, binary_warped.shape, img_w
            )

            # 6. Compute curvature radius (metres)
            curvature = self._compute_curvature(
                left_fit, right_fit, binary_warped.shape
            )

            # 7. Lane width at bottom of warped image
            lane_width = self._compute_lane_width(
                left_fit, right_fit, binary_warped.shape
            )

            # 8. Road detection confidence
            lines_found = (1 if left_fit is not None else 0) + (
                1 if right_fit is not None else 0
            )

            analysis = LaneAnalysis(
                lateral_offset=round(max(-1.0, min(1.0, lateral_offset)), 4),
                lane_width_px=round(lane_width, 1) if lane_width else None,
                road_coverage=float(lines_found / 2),
                road_detected=lines_found >= 1,
                curvature_radius=round(curvature, 1) if curvature else None,
            )

            logger.info(
                f"Lane (Advanced): offset={analysis.lateral_offset:.3f}, "
                f"curvature={analysis.curvature_radius}m, "
                f"lines={lines_found}/2"
            )
            return analysis

        except LaneToolException:
            raise
        except Exception as e:
            raise LaneToolException(f"Advanced lane detection failed: {e}") from e

    # ── Step 1: Thresholding ──────────────────────────────────────────

    def _threshold_binary(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a combined binary image from HLS S-channel and Sobel-X.

        The S-channel of HLS picks up lane markings robustly under
        varying lighting and shadow conditions. Sobel-X on the
        L-channel captures strong vertical edges (lane lines).
        Combining both gives a cleaner binary than Canny alone.
        """
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # ── Sobel-X on L-channel ──
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        max_val = abs_sobelx.max()
        if max_val == 0:
            scaled_sobel = np.zeros_like(l_channel, dtype=np.uint8)
        else:
            scaled_sobel = np.uint8(255 * abs_sobelx / max_val)

        sx_binary = np.zeros_like(scaled_sobel, dtype=np.uint8)
        sx_binary[
            (scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])
        ] = 1

        # ── S-channel threshold ──
        s_binary = np.zeros_like(s_channel, dtype=np.uint8)
        s_binary[
            (s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])
        ] = 1

        # ── Combine ──
        combined = np.zeros_like(s_binary, dtype=np.uint8)
        combined[(sx_binary == 1) | (s_binary == 1)] = 1

        return combined

    # ── Step 2: Perspective Warp ──────────────────────────────────────

    def _perspective_warp(
        self, binary: np.ndarray, img_w: int, img_h: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Warp the binary image to a bird's-eye view.

        Source points are computed dynamically from image dimensions
        using self.src_ratios, so this works for any camera/resolution.

        Returns:
            (warped_binary, M, Minv) — the warp matrices are kept
            for inverse-warping the lane overlay back onto the frame.
        """
        src = np.float32([
            [img_w * r[0], img_h * r[1]] for r in self.src_ratios
        ])

        x_left = img_w * self.dst_x_margin
        x_right = img_w * (1 - self.dst_x_margin)

        dst = np.float32([
            [x_left, img_h * self.dst_y_bottom],   # bottom-left
            [x_left, img_h * self.dst_y_top],       # top-left
            [x_right, img_h * self.dst_y_top],      # top-right
            [x_right, img_h * self.dst_y_bottom],   # bottom-right
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        warped = cv2.warpPerspective(
            binary, M, (img_w, img_h), flags=cv2.INTER_LINEAR
        )
        return warped, M, Minv

    # ── Step 3: Sliding Window Search ─────────────────────────────────

    def _sliding_window_search(
        self, binary_warped: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find lane pixels using a histogram + sliding window approach.

        1. Take a histogram of the bottom half to find left/right peaks.
        2. Slide windows upward, recentering on detected pixels.
        3. Collect all lane pixel coordinates.

        Returns:
            (left_x, left_y, right_x, right_y) — pixel coordinates
            of detected left and right lane pixels.
        """
        h, w = binary_warped.shape

        # Histogram of bottom half
        histogram = np.sum(binary_warped[h // 2 :, :], axis=0)
        midpoint = w // 2

        leftx_base = int(np.argmax(histogram[:midpoint]))
        rightx_base = int(np.argmax(histogram[midpoint:])) + midpoint

        window_height = h // self.nwindows

        # All nonzero pixel coordinates
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for win in range(self.nwindows):
            win_y_low = h - (win + 1) * window_height
            win_y_high = h - win * window_height

            # Left window
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin

            # Right window
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Pixels inside left window
            good_left = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]

            # Pixels inside right window
            good_right = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            # Recenter window on mean of found pixels
            if len(good_left) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        # Concatenate indices
        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

        left_x = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        left_y = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        right_x = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        right_y = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])

        return left_x, left_y, right_x, right_y

    # ── Step 4: Polynomial Fitting ────────────────────────────────────

    def _fit_polynomials(
        self,
        left_x: np.ndarray,
        left_y: np.ndarray,
        right_x: np.ndarray,
        right_y: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Fit a 2nd-degree polynomial (x = ay² + by + c) to each lane.

        We fit x as a function of y because lane lines are more
        vertical than horizontal in the warped image.

        Returns:
            (left_fit, right_fit) — coefficient arrays [a, b, c]
            or None if insufficient pixels were found.
        """
        left_fit = None
        right_fit = None

        if len(left_x) >= 3 and len(left_y) >= 3:
            try:
                left_fit = np.polyfit(left_y, left_x, 2)
            except (np.linalg.LinAlgError, ValueError):
                logger.warning("Failed to fit left lane polynomial")

        if len(right_x) >= 3 and len(right_y) >= 3:
            try:
                right_fit = np.polyfit(right_y, right_x, 2)
            except (np.linalg.LinAlgError, ValueError):
                logger.warning("Failed to fit right lane polynomial")

        return left_fit, right_fit

    # ── Step 5: Lateral Offset ────────────────────────────────────────

    def _compute_lateral_offset(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        warped_shape: Tuple[int, int],
        img_w: int,
    ) -> float:
        """
        Compute how far the vehicle is from the lane centre.

        Convention (same as other lane tools):
            0.0  = centred
           -1.0  = at left lane boundary
           +1.0  = at right lane boundary
        """
        h, w = warped_shape
        img_cx = w / 2
        y_eval = float(h - 1)  # bottom of image

        if left_fit is not None and right_fit is not None:
            left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            lane_cx = (left_x + right_x) / 2
            lane_w = right_x - left_x

            if lane_w < 1:
                return 0.0

            return (img_cx - lane_cx) / (lane_w / 2)

        if left_fit is not None:
            left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            return -1.0 + 2.0 * (img_cx - left_x) / img_cx

        if right_fit is not None:
            right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            return (img_cx - right_x) / (w - img_cx)

        return 0.0

    # ── Step 6: Curvature ─────────────────────────────────────────────

    def _compute_curvature(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        warped_shape: Tuple[int, int],
    ) -> Optional[float]:
        """
        Compute the radius of curvature in metres.

        Uses the standard formula:
            R = (1 + (2Ay + B)²)^(3/2) / |2A|

        applied to polynomials re-fitted in real-world (metre) space.
        Returns the average of left and right curvature, or whichever
        is available. None if no lanes detected.
        """
        if left_fit is None and right_fit is None:
            return None

        h = warped_shape[0]
        ploty = np.linspace(0, h - 1, h)
        y_eval = float(h - 1)

        curvatures = []

        for fit in [left_fit, right_fit]:
            if fit is None:
                continue

            # Reconstruct x values in pixel space
            fitx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]

            # Re-fit in real-world coordinates
            fit_cr = np.polyfit(
                ploty * self.ym_per_pix,
                fitx * self.xm_per_pix,
                2,
            )

            # Curvature formula
            A, B = fit_cr[0], fit_cr[1]
            y_m = y_eval * self.ym_per_pix
            curverad = ((1 + (2 * A * y_m + B) ** 2) ** 1.5) / max(
                abs(2 * A), 1e-6
            )
            curvatures.append(curverad)

        if not curvatures:
            return None

        avg_curvature = sum(curvatures) / len(curvatures)

        # Cap at 10 km to avoid absurdly large values on straight roads
        return min(avg_curvature, 10000.0)

    # ── Step 7: Lane Width ────────────────────────────────────────────

    def _compute_lane_width(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        warped_shape: Tuple[int, int],
    ) -> Optional[float]:
        """Lane width in pixels at the bottom of the warped image."""
        if left_fit is None or right_fit is None:
            return None

        y_eval = float(warped_shape[0] - 1)
        left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
        right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]

        width = abs(right_x - left_x)
        return width if width > 1 else None


# ── Visual debugging ──────────────────────────────────────────────────────

if __name__ == "__main__":

    tool = LaneToolAdvanced()

    BASE_DIR = os.path.dirname(__file__)
    img_path = os.path.join(BASE_DIR, "images", "test_img2.jpg")

    # Run analysis
    analysis = tool.detect_lanes(img_path)
    print("\n===== LANE ANALYSIS (Advanced) =====")
    print("Lateral offset:", analysis.lateral_offset)
    print("Lane width:", analysis.lane_width_px)
    print("Road coverage:", analysis.road_coverage)
    print("Road detected:", analysis.road_detected)
    print("Curvature radius:", analysis.curvature_radius, "m")

    # ── Visualization ─────────────────────────────────────────────────
    frame = cv2.imread(img_path)
    img_h, img_w = frame.shape[:2]

    # Show the thresholded binary
    binary = tool._threshold_binary(frame)
    binary_vis = (binary * 255).astype(np.uint8)

    # Show the warped binary
    warped, M, Minv = tool._perspective_warp(binary, img_w, img_h)
    warped_vis = (warped * 255).astype(np.uint8)

    # Find lane pixels and fit
    lx, ly, rx, ry = tool._sliding_window_search(warped)
    left_fit, right_fit = tool._fit_polynomials(lx, ly, rx, ry)

    # Draw lane overlay on warped image
    warped_color = cv2.cvtColor(warped_vis, cv2.COLOR_GRAY2BGR)
    if len(lx) > 0:
        for x, y in zip(lx, ly):
            cv2.circle(warped_color, (int(x), int(y)), 1, (255, 0, 0), -1)
    if len(rx) > 0:
        for x, y in zip(rx, ry):
            cv2.circle(warped_color, (int(x), int(y)), 1, (0, 0, 255), -1)

    # Draw fitted polynomials
    ploty = np.linspace(0, img_h - 1, img_h)
    if left_fit is not None:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        pts = np.column_stack((left_fitx.astype(int), ploty.astype(int)))
        cv2.polylines(warped_color, [pts], False, (0, 255, 255), 2)
    if right_fit is not None:
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        pts = np.column_stack((right_fitx.astype(int), ploty.astype(int)))
        cv2.polylines(warped_color, [pts], False, (0, 255, 255), 2)

    # Warp lane overlay back onto original frame
    if left_fit is not None and right_fit is not None:
        warp_zero = np.zeros_like(warped, dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.column_stack((left_fitx, ploty)).astype(np.int32)
        pts_right = np.flipud(np.column_stack((right_fitx, ploty))).astype(np.int32)
        pts_all = np.vstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, [pts_all], (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (img_w, img_h))
        result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    else:
        result = frame.copy()

    cv2.putText(
        result,
        f"Offset: {analysis.lateral_offset:.4f}  Curvature: {analysis.curvature_radius}m",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )

    cv2.imshow("Binary Threshold", binary_vis)
    cv2.imshow("Warped (Bird's Eye)", warped_color)
    cv2.imshow("Lane Overlay", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()