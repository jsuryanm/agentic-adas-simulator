import os
import tempfile

import cv2
import requests
import streamlit as st

from src.core.config import settings
from src.tools.video_tool import annotate_image

# ── Page config ───────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic ADAS Simulator",
    page_icon="🚗",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────

DECISION_ORDER = {"SAFE": 0, "ADVISORY": 1, "WARNING": 2, "CRITICAL": 3}

RISK_EMOJI = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🟠",
    "critical": "🔴",
}

DECISION_ICON = {
    "SAFE": "✅",
    "ADVISORY": "⚠️",
    "WARNING": "🚨",
    "CRITICAL": "🛑",
}


# ── API helpers ───────────────────────────────────────────────────────────

def check_backend() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except requests.ConnectionError:
        return False


def api_analyse_image(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        resp = requests.post(f"{API_URL}/analyse/image", files={"file": f})
    resp.raise_for_status()
    return resp.json()


def api_analyse_video(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        resp = requests.post(f"{API_URL}/analyse/video", files={"file": f})
    resp.raise_for_status()
    return resp.json()


def api_download_video(filename: str) -> str | None:
    resp = requests.get(f"{API_URL}/download/video/{filename}", stream=True)
    if resp.status_code != 200:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        for chunk in resp.iter_content(chunk_size=8_192):
            tmp.write(chunk)
        return tmp.name


# ── Render helpers ────────────────────────────────────────────────────────

def render_decision(decision: dict):
    """Colored decision banner using Streamlit native callouts."""
    dec_type = decision.get("decision_type", "SAFE")
    rec = decision.get("recommendation", "")
    conf = decision.get("confidence", 0)
    icon = DECISION_ICON.get(dec_type, "ℹ️")
    body = f"**{dec_type}** — {rec}  \n_Confidence: {conf:.0%}_"

    if dec_type == "SAFE":
        st.success(f"{icon} {body}")
    elif dec_type == "ADVISORY":
        st.info(f"{icon} {body}")
    elif dec_type == "WARNING":
        st.warning(f"{icon} {body}")
    else:  # CRITICAL
        st.error(f"{icon} {body}")


def render_pipeline_timing(processing_time: dict):
    """Show per-agent timing as compact metrics."""
    agents = [
        ("Perception", "perception_agent"),
        ("Scene", "scene_agent"),
        ("Risk", "risk_agent"),
        ("Decision", "decision_agent"),
    ]
    cols = st.columns(len(agents))
    for col, (label, key) in zip(cols, agents):
        t = processing_time.get(key)
        col.metric(label, f"{t:.2f}s" if t else "—")


def render_risk_report(risk: dict):
    st.subheader("Risk Assessment")
    cols = st.columns(4)
    for col, (label, key) in zip(
        cols,
        [("Collision", "collision_risk"),
         ("Pedestrian", "pedestrian_risk"),
         ("Lane", "lane_risk"),
         ("Composite", "composite_risk")],
    ):
        level = risk.get(key, "low")
        col.metric(label, f"{RISK_EMOJI.get(level, '')} {level.upper()}")

    explanation = risk.get("explanation", "")
    if explanation:
        st.caption(explanation)


def render_scene_summary(scene: dict):
    st.subheader("Scene Summary")

    lead_present = scene.get("lead_vehicle_present", False)
    lead_dist = scene.get("lead_vehicle_distance", "—")
    ped_present = scene.get("pedestrian_present", False)
    ped_near = scene.get("pedestrian_near_path", False)
    traffic = scene.get("traffic_density", "unknown")
    lane = scene.get("lane_status", "centered").replace("_", " ")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lead Vehicle", f"{lead_dist}" if lead_present else "None")
    col2.metric("Pedestrian", "In path ⚠️" if ped_near else ("Visible" if ped_present else "Clear"))
    col3.metric("Traffic", traffic.capitalize())
    col4.metric("Lane", lane.title())

    narration = scene.get("llm_narration", "")
    if narration:
        st.info(f"💬 {narration}")

    notes = scene.get("context_notes", [])
    if notes:
        with st.expander("Context notes", expanded=True):
            for note in notes:
                st.markdown(f"- {note}")


def render_detections(objects: list):
    st.subheader(f"Detected Objects ({len(objects)})")

    if not objects:
        st.caption("No objects detected.")
        return

    # Count summary
    counts: dict[str, int] = {}
    for obj in objects:
        label = obj.get("label", "unknown")
        counts[label] = counts.get(label, 0) + 1

    st.caption("  ".join(f"**{label}** ×{n}" for label, n in sorted(counts.items())))

    # Individual detections
    for obj in objects:
        label = obj.get("label", "?")
        conf = obj.get("confidence", 0)
        pos = obj.get("position", "?")
        dist = obj.get("distance", "?")
        depth = obj.get("depth_value")
        depth_str = f"  depth={depth:.2f}" if depth is not None else ""
        st.text(f"  {label:<14}  {conf:.0%}  {pos:<8}  {dist}{depth_str}")


def render_annotated_image(image_path: str, result: dict):
    try:
        annotated = annotate_image(image_path, result)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    except Exception as e:
        st.warning(f"Annotation failed: {e}")
        st.image(image_path, use_container_width=True)


def show_image_results(result: dict, image_path: str):
    """Full results panel for a single image."""
    decision = result.get("decision", {})
    risk = result.get("risk_report", {})
    scene = result.get("scene_summary", {})
    objects = result.get("detected_objects", [])
    timing = result.get("processing_time", {})
    errors = result.get("errors", [])

    render_decision(decision)

    st.divider()
    render_pipeline_timing(timing)

    st.divider()
    col_img, col_detail = st.columns([3, 2])

    with col_img:
        st.subheader("Annotated Frame")
        render_annotated_image(image_path, result)

    with col_detail:
        render_risk_report(risk)
        st.divider()
        render_scene_summary(scene)

    st.divider()
    render_detections(objects)

    explanation = decision.get("explanation", "")
    if explanation:
        st.divider()
        st.subheader("Reasoning Chain")
        st.markdown(explanation)

    if errors:
        with st.expander("⚠️ Pipeline Errors", expanded=False):
            for err in errors:
                st.error(err)

    total = sum(timing.values()) if timing else 0
    st.caption(f"Total pipeline time: {total:.2f}s")


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚗 ADAS Simulator")
    st.caption("Agentic Driver Assistance Simulator")

    backend_ok = check_backend()
    if backend_ok:
        st.success("Backend connected", icon="✅")
    else:
        st.error(
            "Backend unreachable.\n\nStart it with:\n```\nuv run uvicorn backend.app:app\n```",
            icon="🔴",
        )

    st.divider()
    input_mode = st.radio("Input mode", ["Image", "Video"])

    st.divider()
    st.subheader("Settings")

    confidence = st.slider("Detection confidence", 0.1, 1.0, settings.CONFIDENCE_THRESHOLD, 0.05)
    depth_enabled = st.toggle("Depth estimation", value=settings.DEPTH_ENABLED)
    llm_enabled = st.toggle("LLM reasoning", value=settings.LLM_ENABLED)

    if input_mode == "Video":
        sample_rate = st.slider(
            "Sample interval (sec)", 0.5, 5.0, settings.SECONDS_PER_SAMPLE, 0.5
        )
        settings.SECONDS_PER_SAMPLE = sample_rate

    settings.CONFIDENCE_THRESHOLD = confidence
    settings.DEPTH_ENABLED = depth_enabled
    settings.LLM_ENABLED = llm_enabled

    st.divider()
    st.caption("Pipeline: Perception → Scene → Risk → Decision")
    st.caption("Models: YOLOv8 · Depth Anything V2 · GPT-4o-mini")


# ── Main area ─────────────────────────────────────────────────────────────

st.title("Agentic ADAS Simulator")
st.caption("Multi-agent driving scene analysis — upload an image or video to begin.")

# ── Image mode ────────────────────────────────────────────────────────────

if input_mode == "Image":
    uploaded = st.file_uploader(
        "Upload a driving scene",
        type=["jpg", "jpeg", "png"],
        help="Dashcam frame or driving scene photo",
    )

    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        col_prev, col_info = st.columns([3, 1])
        with col_prev:
            st.image(tmp_path, caption="Input frame", use_container_width=True)
        with col_info:
            st.metric("File", uploaded.name)
            st.metric("Size", f"{uploaded.size / 1024:.1f} KB")

        if st.button("▶ Run Analysis", type="primary", disabled=not backend_ok,
                     use_container_width=True):
            try:
                with st.spinner("Running ADAS pipeline..."):
                    result = api_analyse_image(tmp_path)
                st.session_state["image_result"] = result
                st.session_state["image_path"] = tmp_path
            except requests.HTTPError as e:
                st.error(f"Backend error: {e}")
            except requests.ConnectionError:
                st.error("Lost connection to backend.")

        if "image_result" in st.session_state and "image_path" in st.session_state:
            st.divider()
            show_image_results(
                st.session_state["image_result"],
                st.session_state["image_path"],
            )

# ── Video mode ────────────────────────────────────────────────────────────

elif input_mode == "Video":
    uploaded = st.file_uploader(
        "Upload a dashcam video",
        type=["mp4", "avi", "mov"],
        help="Short dashcam clip — 10–30 seconds recommended",
    )

    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(tmp_path)
        st.caption(
            f"{uploaded.name}  ·  {uploaded.size / (1024 * 1024):.1f} MB  "
            f"·  sampling every {settings.SECONDS_PER_SAMPLE:.1f}s"
        )

        if st.button("▶ Run Video Analysis", type="primary", disabled=not backend_ok,
                     use_container_width=True):
            try:
                with st.spinner("Processing video — this may take a while..."):
                    result = api_analyse_video(tmp_path)
                st.session_state["video_result"] = result
            except requests.HTTPError as e:
                st.error(f"Backend error: {e}")
            except requests.ConnectionError:
                st.error("Lost connection to backend.")

        if "video_result" in st.session_state:
            result = st.session_state["video_result"]
            frames = result.get("frames", [])
            output_video_path = result.get("output_video")

            st.divider()

            # ── Summary metrics ───────────────────────────────────────
            decisions = [f.get("decision", {}).get("decision_type", "SAFE") for f in frames]
            worst = max(decisions, key=lambda d: DECISION_ORDER.get(d, 0), default="SAFE")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Frames Analysed", len(frames))
            m2.metric("Peak Severity", worst)
            m3.metric("Critical Frames", decisions.count("CRITICAL"))
            m4.metric("Warning Frames", decisions.count("WARNING"))

            # ── Annotated output video ────────────────────────────────
            if output_video_path:
                local_video = api_download_video(os.path.basename(output_video_path))
                if local_video:
                    st.subheader("Annotated Output Video")
                    st.video(local_video)

            # ── Decision timeline ─────────────────────────────────────
            st.subheader("Decision Timeline")

            color_map = {
                "SAFE": "green",
                "ADVISORY": "blue",
                "WARNING": "orange",
                "CRITICAL": "red",
            }

            # One progress-style bar built from individual colored spans
            bars_html = "".join(
                f'<div title="Frame {f["frame_number"]} — t={f.get("timestamp",0):.1f}s — '
                f'{f.get("decision",{}).get("decision_type","SAFE")}" '
                f'style="flex:1;height:28px;background:{color_map.get(f.get("decision",{}).get("decision_type","SAFE"),"gray")};'
                f'opacity:0.8;border-radius:2px;cursor:pointer;"></div>'
                for f in frames
            )
            st.markdown(
                f'<div style="display:flex;gap:2px;">{bars_html}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "🟢 SAFE  🔵 ADVISORY  🟠 WARNING  🔴 CRITICAL"
            )

            # ── Frame inspector ───────────────────────────────────────
            if frames:
                st.divider()
                st.subheader("Frame Inspector")

                idx = st.slider("Select frame", 0, len(frames) - 1, 0,
                                format="Frame %d")

                frame_data = frames[idx]
                ts = frame_data.get("timestamp", 0)
                frame_num = frame_data.get("frame_number", idx)
                obj_count = frame_data.get("detected_objects_count", "?")

                st.caption(
                    f"Frame {frame_num}  ·  t = {ts:.1f}s  ·  {obj_count} objects"
                )

                render_decision(frame_data.get("decision", {}))

                frame_risk = frame_data.get("risk_report", {})
                if frame_risk:
                    render_risk_report(frame_risk)