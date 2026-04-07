import os
import time
import tempfile
import shutil
import subprocess

import cv2
import numpy as np
from src.logger.custom_logger import logger 

from src.core.config import settings
from src.pipelines.graph import run_pipeline
from src.exceptions.custom_exceptions import ImageLoadException


# Colour scheme for risk levels
RISK_COLOURS = {
    "low": (0, 200, 0),        # green
    "medium": (0, 200, 255),   # orange
    "high": (0, 100, 255),     # red-orange
    "critical": (0, 0, 255),   # red
}

DECISION_COLOURS = {
    "SAFE": (0, 200, 0),
    "ADVISORY": (0, 200, 255),
    "WARNING": (0, 100, 255),
    "CRITICAL": (0, 0, 255),
}

LABEL_COLOURS = {
    "car": (255, 255, 255),
    "truck": (200, 200, 200),
    "bus": (200, 200, 200),
    "motorcycle": (0, 200, 200),
    "rider": (0, 255, 255),
    "cyclist": (255, 165, 0),
    "pedestrian": (255, 0, 0),
    "traffic_light": (0, 255, 0),
    "stop_sign": (0, 0, 255),
}


# Frame extraction 
def extract_frames(video_path: str, seconds_per_sample: float = None) -> list:
    """Extract frames from a video at the configured sampling rate.

    Args:
        video_path: Path to the input video file.
        seconds_per_sample: Override for config.SECONDS_PER_SAMPLE.

    Returns:
        List of dicts: [{"path": str, "frame_number": int, "timestamp": float}]
    """
    if seconds_per_sample is None:
        seconds_per_sample = settings.SECONDS_PER_SAMPLE

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ImageLoadException(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video: {fps:.1f} fps, {total_frames} frames, {duration:.1f}s")

    frame_interval = max(1, int(fps * seconds_per_sample))

    temp_dir = tempfile.mkdtemp(prefix="adas_frames_")
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps if fps > 0 else 0
            filename = f"frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(temp_dir, filename)
            cv2.imwrite(filepath, frame)

            frames.append({
                "path": filepath,
                "frame_number": frame_idx,
                "timestamp": timestamp,
            })

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames to {temp_dir}")
    return frames


# Video processing 

def process_video(video_path: str,
                  seconds_per_sample: float = None,
                  progress_callback=None) -> dict:
    """Run the ADAS pipeline on each sampled frame of a video.

    Args:
        video_path: Path to input video.
        seconds_per_sample: Sampling rate override.
        progress_callback: Optional callable(current, total) for UI updates.

    Returns:
        dict with "frames" (list of results) and "output_video" (path).
    """
    frames = extract_frames(video_path, seconds_per_sample)
    results = []

    for i, frame_info in enumerate(frames):
        logger.info(f"Processing frame {i + 1}/{len(frames)} "
                     f"(t={frame_info['timestamp']:.1f}s)")

        state = run_pipeline(frame_info["path"], frame_info["frame_number"])

        results.append({
            "frame_number": frame_info["frame_number"],
            "timestamp": frame_info["timestamp"],
            "image_path": frame_info["path"],
            "state": state,
        })

        if progress_callback:
            progress_callback(i + 1, len(frames))

    # Generate annotated output video
    output_path = _create_output_video(video_path, results)

    # Clean up temp frames
    temp_dir = os.path.dirname(frames[0]["path"]) if frames else None
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"frames": results, "output_video": output_path}


#  Annotated output video 

def _create_output_video(original_video_path: str,
                         results: list) -> str:
    """Create an annotated video showing detections, risk, and decisions.

    For each sampled frame, the annotation is held until the next sample
    so the output video plays at the original frame rate with overlays.
    """
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise ImageLoadException(f"Cannot reopen video: {original_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(settings.VIDEO_OUTPUT_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(original_video_path))[0]

    # Write to a temporary file first (mp4v codec — not browser-compatible)
    temp_output = os.path.join(settings.VIDEO_OUTPUT_DIR,
                               f"{base_name}_adas_temp.mp4")
    final_output = os.path.join(settings.VIDEO_OUTPUT_DIR,
                                f"{base_name}_adas_output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # Build a lookup: frame_number → result state
    result_lookup = {}
    for r in results:
        result_lookup[r["frame_number"]] = r["state"]

    # Get sorted sample frame numbers for "hold last annotation" logic
    sample_frames = sorted(result_lookup.keys())

    current_state = None
    sample_idx = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update annotation when we reach the next sampled frame
        if sample_idx < len(sample_frames) and frame_idx >= sample_frames[sample_idx]:
            current_state = result_lookup[sample_frames[sample_idx]]
            sample_idx += 1

        # Draw annotations on every frame using the current state
        if current_state is not None:
            frame = _annotate_frame(frame, current_state)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Re-encode to H.264 for browser compatibility
    final_output = _reencode_to_h264(temp_output, final_output)

    logger.info(f"Output video saved: {final_output}")
    return final_output


def _reencode_to_h264(input_path: str, output_path: str) -> str:
    """Re-encode an mp4v video to H.264 using ffmpeg.

    Falls back to the original mp4v file if ffmpeg is not available.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",  # required for browser playback
            "-movflags", "+faststart",  # enables streaming before full download
            "-an",  # no audio track
            output_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )

        if result.returncode == 0 and os.path.exists(output_path):
            # Remove the temp mp4v file
            os.remove(input_path)
            logger.info("Video re-encoded to H.264 successfully")
            return output_path
        else:
            logger.warning(f"ffmpeg failed (code {result.returncode}), "
                           f"using mp4v fallback")
            return input_path

    except FileNotFoundError:
        logger.warning("ffmpeg not found — output video will use mp4v codec "
                       "(may not play in browsers)")
        return input_path

    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out — using mp4v fallback")
        return input_path


#  Frame annotation 

def _annotate_frame(frame: np.ndarray, state: dict) -> np.ndarray:
    """Draw bounding boxes, risk panel, and decision on a single frame."""
    annotated = frame.copy()
    img_h, img_w = annotated.shape[:2]

    # 1. Draw bounding boxes for detected objects
    for obj in state.get("detected_objects", []):
        bbox = obj.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        label = obj.get("label", "?")
        conf = obj.get("confidence", 0)
        dist = obj.get("distance", "?")
        colour = LABEL_COLOURS.get(label, (255, 255, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        text = f"{label} {conf:.0%} [{dist}]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 2. Risk panel (top-right corner)
    risk_report = state.get("risk_report", {})
    decision = state.get("decision", {})

    panel_w, panel_h = 280, 150
    px, py = img_w - panel_w - 10, 10

    # Semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)

    y_offset = py + 20
    line_gap = 22

    # Risk lines
    for label_key, display in [("collision_risk", "Collision"),
                                ("pedestrian_risk", "Pedestrian"),
                                ("lane_risk", "Lane"),
                                ("composite_risk", "COMPOSITE")]:
        level = risk_report.get(label_key, "low")
        colour = RISK_COLOURS.get(level, (200, 200, 200))
        text = f"{display}: {level.upper()}"

        if label_key == "composite_risk":
            cv2.putText(annotated, text, (px + 8, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        else:
            cv2.putText(annotated, text, (px + 8, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        y_offset += line_gap

    # 3. Decision banner (top-left)
    dec_type = decision.get("decision_type", "SAFE")
    dec_colour = DECISION_COLOURS.get(dec_type, (200, 200, 200))
    rec = decision.get("recommendation", "")

    banner_text = f"DECISION: {dec_type}"
    (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    cv2.rectangle(annotated, (10, 10), (bw + 24, bh + 24), dec_colour, -1)
    cv2.putText(annotated, banner_text, (16, bh + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Recommendation text below banner
    if rec:
        max_chars = 60
        short_rec = rec[:max_chars] + "..." if len(rec) > max_chars else rec
        cv2.putText(annotated, short_rec, (16, bh + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 4. Lane status (bottom-center)
    scene = state.get("scene_summary", {})
    lane_status = scene.get("lane_status", "centered")
    lane_text = f"Lane: {lane_status.replace('_', ' ').title()}"
    (lw, lh), _ = cv2.getTextSize(lane_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    lx = (img_w - lw) // 2
    ly = img_h - 20
    cv2.putText(annotated, lane_text, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


#  Single image annotation (for dashboard) 

def annotate_image(image_path: str, state: dict) -> np.ndarray:
    """Annotate a single image with pipeline results for display."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ImageLoadException(f"Cannot read: {image_path}")
    return _annotate_frame(frame, state)