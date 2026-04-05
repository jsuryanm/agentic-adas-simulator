import os
import tempfile
import requests
import cv2
import streamlit as st

from src.core.config import settings
from src.tools.video_tool import annotate_image  # display-only CV drawing utility

# ── Page config ───────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic ADAS Simulator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --adas-green: #00E676;
    --adas-amber: #FFB300;
    --adas-orange: #FF6D00;
    --adas-red: #FF1744;
}

.stApp {
    background: linear-gradient(135deg, #0A0E17 0%, #0F172A 50%, #0A0E17 100%);
}

section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E293B !important;
}

/* Decision banner */
.decision-banner {
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.decision-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.decision-SAFE {
    background: linear-gradient(135deg, rgba(0,230,118,0.08) 0%, rgba(0,230,118,0.02) 100%);
    border-color: rgba(0,230,118,0.3);
}
.decision-SAFE::before { background: #00E676; }
.decision-ADVISORY {
    background: linear-gradient(135deg, rgba(255,179,0,0.08) 0%, rgba(255,179,0,0.02) 100%);
    border-color: rgba(255,179,0,0.3);
}
.decision-ADVISORY::before { background: #FFB300; }
.decision-WARNING {
    background: linear-gradient(135deg, rgba(255,109,0,0.08) 0%, rgba(255,109,0,0.02) 100%);
    border-color: rgba(255,109,0,0.3);
}
.decision-WARNING::before { background: #FF6D00; }
.decision-CRITICAL {
    background: linear-gradient(135deg, rgba(255,23,68,0.08) 0%, rgba(255,23,68,0.02) 100%);
    border-color: rgba(255,23,68,0.3);
    animation: critical-pulse 2s ease-in-out infinite;
}
.decision-CRITICAL::before { background: #FF1744; }

@keyframes critical-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(255,23,68,0.1); }
    50% { box-shadow: 0 0 30px rgba(255,23,68,0.25); }
}

.decision-type {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.decision-rec {
    font-size: 0.95rem;
    font-family: 'DM Sans', sans-serif;
    opacity: 0.85;
    line-height: 1.5;
}

/* Risk pills */
.risk-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.risk-low { background: rgba(0,230,118,0.15); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }
.risk-medium { background: rgba(255,179,0,0.15); color: #FFB300; border: 1px solid rgba(255,179,0,0.3); }
.risk-high { background: rgba(255,109,0,0.15); color: #FF6D00; border: 1px solid rgba(255,109,0,0.3); }
.risk-critical { background: rgba(255,23,68,0.15); color: #FF1744; border: 1px solid rgba(255,23,68,0.3); }

/* Section headers */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748B;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 1rem;
}

/* Detection cards */
.detection-item {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.detection-label { font-weight: 600; color: #E2E8F0; text-transform: capitalize; }
.detection-meta {
    color: #94A3B8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}

/* Context notes */
.context-note {
    background: rgba(56, 189, 248, 0.06);
    border-left: 3px solid #38BDF8;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.4rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #CBD5E1;
    font-family: 'DM Sans', sans-serif;
}

/* Pipeline stages */
.pipeline-stages {
    display: flex;
    gap: 0;
    margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
}
.pipeline-stage {
    flex: 1;
    text-align: center;
    padding: 0.6rem 0.4rem;
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid #1E293B;
    color: #64748B;
}
.pipeline-stage:first-child { border-radius: 8px 0 0 8px; }
.pipeline-stage:last-child { border-radius: 0 8px 8px 0; }
.pipeline-stage.done {
    background: rgba(0, 230, 118, 0.08);
    border-color: rgba(0, 230, 118, 0.2);
    color: #00E676;
}

/* Scene grid */
.scene-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
.scene-item {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
}
.scene-item-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.scene-item-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #E2E8F0;
    font-weight: 500;
    margin-top: 0.15rem;
}

/* Title */
.main-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: #64748B;
    margin-bottom: 2rem;
}

/* Misc */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 0.8rem;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────

DECISION_COLORS = {
    "SAFE": "#00E676",
    "ADVISORY": "#FFB300",
    "WARNING": "#FF6D00",
    "CRITICAL": "#FF1744",
}

DEC_ORDER = {"SAFE": 0, "ADVISORY": 1, "WARNING": 2, "CRITICAL": 3}


# ── API helpers ───────────────────────────────────────────────────────────

def check_backend() -> bool:
    """Verify the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def api_analyse_image(file_path: str) -> dict:
    """POST image to /analyse/image and return the JSON response."""
    with open(file_path, "rb") as f:
        resp = requests.post(f"{API_URL}/analyse/image", files={"file": f})
    resp.raise_for_status()
    return resp.json()


def api_analyse_video(file_path: str) -> dict:
    """POST video to /analyse/video and return the JSON response."""
    with open(file_path, "rb") as f:
        resp = requests.post(f"{API_URL}/analyse/video", files={"file": f})
    resp.raise_for_status()
    return resp.json()


def api_download_video(filename: str) -> str | None:
    """Download annotated video from backend and return local temp path."""
    resp = requests.get(f"{API_URL}/download/video/{filename}", stream=True)
    if resp.status_code != 200:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        return tmp.name


# ── Render helpers ────────────────────────────────────────────────────────

def risk_pill(level: str) -> str:
    level = (level or "low").lower()
    return f'<span class="risk-pill risk-{level}">{level}</span>'


def render_decision_banner(decision: dict):
    dec_type = decision.get("decision_type", "SAFE")
    rec = decision.get("recommendation", "")
    conf = decision.get("confidence", 0)
    color = DECISION_COLORS.get(dec_type, "#94A3B8")

    st.markdown(f"""
    <div class="decision-banner decision-{dec_type}">
        <div class="decision-type" style="color: {color};">{dec_type}</div>
        <div class="decision-rec">{rec}</div>
        <div style="margin-top: 0.6rem; font-family: 'JetBrains Mono', monospace;
                    font-size: 0.75rem; color: #64748B;">
            CONFIDENCE {conf:.0%}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_stages(processing_time: dict):
    stages = [
        ("PERCEPTION", "perception_agent"),
        ("SCENE", "scene_agent"),
        ("RISK", "risk_agent"),
        ("DECISION", "decision_agent"),
    ]
    html = '<div class="pipeline-stages">'
    for label, key in stages:
        t = processing_time.get(key)
        cls = "done" if t is not None else ""
        time_str = (
            f"<br><span style='font-size:0.65rem;opacity:0.7;'>{t:.2f}s</span>"
            if t else ""
        )
        html += f'<div class="pipeline-stage {cls}">{label}{time_str}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_risk_report(risk: dict):
    st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for col, (label, key) in zip(
        cols,
        [("Collision", "collision_risk"), ("Pedestrian", "pedestrian_risk"),
         ("Lane", "lane_risk"), ("Composite", "composite_risk")],
    ):
        level = risk.get(key, "low")
        with col:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                            color: #64748B; letter-spacing: 0.1em; margin-bottom: 0.4rem;">
                    {label.upper()}
                </div>
                {risk_pill(level)}
            </div>
            """, unsafe_allow_html=True)

    explanation = risk.get("explanation", "")
    if explanation:
        st.markdown(f"""
        <div style="margin-top: 1rem; padding: 0.8rem; background: rgba(30,41,59,0.3);
                    border-radius: 8px; border: 1px solid #1E293B;
                    font-family: 'DM Sans', sans-serif; font-size: 0.85rem; color: #94A3B8;">
            {explanation}
        </div>
        """, unsafe_allow_html=True)


def render_scene_summary(scene: dict):
    st.markdown('<div class="section-header">Scene Understanding</div>', unsafe_allow_html=True)

    lead_present = scene.get("lead_vehicle_present", False)
    lead_dist = scene.get("lead_vehicle_distance", "—")
    ped_present = scene.get("pedestrian_present", False)
    ped_near = scene.get("pedestrian_near_path", False)
    traffic = scene.get("traffic_density", "unknown")
    lane = scene.get("lane_status", "centered")

    lead_text = f"Yes — {lead_dist} range" if lead_present else "None detected"
    ped_text = "In path" if ped_near else ("Visible" if ped_present else "Clear")

    st.markdown(f"""
    <div class="scene-grid">
        <div class="scene-item">
            <div class="scene-item-label">Lead Vehicle</div>
            <div class="scene-item-value">{lead_text}</div>
        </div>
        <div class="scene-item">
            <div class="scene-item-label">Pedestrian</div>
            <div class="scene-item-value">{ped_text}</div>
        </div>
        <div class="scene-item">
            <div class="scene-item-label">Traffic Density</div>
            <div class="scene-item-value" style="text-transform: capitalize;">{traffic}</div>
        </div>
        <div class="scene-item">
            <div class="scene-item-label">Lane Status</div>
            <div class="scene-item-value" style="text-transform: capitalize;">{lane.replace("_", " ")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    narration = scene.get("llm_narration", "")
    if narration:
        st.markdown(f"""
        <div style="margin-top: 0.8rem; padding: 0.8rem; background: rgba(56,189,248,0.04);
                    border: 1px solid rgba(56,189,248,0.15); border-radius: 8px;
                    font-family: 'DM Sans', sans-serif; font-size: 0.85rem;
                    color: #CBD5E1; font-style: italic;">
            "{narration}"
        </div>
        """, unsafe_allow_html=True)

    notes = scene.get("context_notes", [])
    if notes:
        for note in notes:
            st.markdown(f'<div class="context-note">{note}</div>', unsafe_allow_html=True)


def render_detections(detected_objects: list):
    st.markdown('<div class="section-header">Detected Objects</div>', unsafe_allow_html=True)

    if not detected_objects:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #64748B;
                    font-family: 'DM Sans', sans-serif;">
            No objects detected
        </div>
        """, unsafe_allow_html=True)
        return

    # Summary chips
    label_counts: dict[str, int] = {}
    for obj in detected_objects:
        label = obj.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1

    chips = " ".join(
        f'<span style="background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.2);'
        f' padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.78rem;'
        f' font-family: JetBrains Mono, monospace; color: #38BDF8;">'
        f'{count}x {label}</span>'
        for label, count in sorted(label_counts.items())
    )
    st.markdown(f'<div style="margin-bottom: 0.8rem;">{chips}</div>', unsafe_allow_html=True)

    for obj in detected_objects:
        label = obj.get("label", "?")
        conf = obj.get("confidence", 0)
        pos = obj.get("position", "?")
        dist = obj.get("distance", "?")
        depth = obj.get("depth_value")
        depth_str = f" · d={depth:.2f}" if depth is not None else ""

        st.markdown(f"""
        <div class="detection-item">
            <span class="detection-label">{label}</span>
            <span class="detection-meta">{conf:.0%} · {pos} · {dist}{depth_str}</span>
        </div>
        """, unsafe_allow_html=True)


def render_annotated_image(image_path: str, result: dict):
    """Draw bounding boxes and overlays on the uploaded image using API results."""
    try:
        annotated = annotate_image(image_path, result)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not annotate image: {e}")
        st.image(image_path, use_container_width=True)


def show_image_results(result: dict, image_path: str):
    """Render the full results panel for a single image."""
    decision = result.get("decision", {})
    risk = result.get("risk_report", {})
    scene = result.get("scene_summary", {})
    objects = result.get("detected_objects", [])
    timing = result.get("processing_time", {})
    errors = result.get("errors", [])

    render_pipeline_stages(timing)
    render_decision_banner(decision)

    # Two-column: annotated image + risk/scene
    col_img, col_detail = st.columns([3, 2])

    with col_img:
        st.markdown('<div class="section-header">Annotated Frame</div>',
                    unsafe_allow_html=True)
        render_annotated_image(image_path, result)

    with col_detail:
        render_risk_report(risk)
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        render_scene_summary(scene)

    # Detections full-width
    render_detections(objects)

    # Reasoning chain
    explanation = decision.get("explanation", "")
    if explanation:
        st.markdown('<div class="section-header">Reasoning Chain</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding: 1rem; background: rgba(30,41,59,0.3); border: 1px solid #1E293B;
                    border-radius: 8px; font-family: 'DM Sans', sans-serif;
                    font-size: 0.9rem; color: #CBD5E1; line-height: 1.6;">
            {explanation}
        </div>
        """, unsafe_allow_html=True)

    if errors:
        with st.expander("Pipeline Errors", expanded=False):
            for err in errors:
                st.error(err)

    total = sum(timing.values()) if timing else 0
    st.markdown(f"""
    <div style="text-align: center; margin-top: 1.5rem; padding-top: 1rem;
                border-top: 1px solid #1E293B; font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem; color: #475569;">
        Total pipeline time: {total:.2f}s
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem;
                font-weight: 700; color: #F1F5F9; margin-bottom: 0.2rem;">
        ADAS SIM
    </div>
    <div style="font-family: 'DM Sans', sans-serif; font-size: 0.78rem;
                color: #64748B; margin-bottom: 1.5rem;">
        Agentic Driver Assistance Simulator
    </div>
    """, unsafe_allow_html=True)

    # Backend status
    backend_ok = check_backend()
    if backend_ok:
        st.success("Backend connected", icon="✅")
    else:
        st.error("Backend unreachable — start it with:\n\n"
                 "`uv run uvicorn backend.app:app`", icon="🔴")

    st.markdown("---")
    input_mode = st.radio("Input Mode", ["Image", "Video"], horizontal=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
                letter-spacing: 0.1em; color: #64748B; margin-bottom: 0.8rem;">
        PIPELINE SETTINGS
    </div>
    """, unsafe_allow_html=True)

    confidence = st.slider(
        "Detection confidence", 0.1, 1.0,
        settings.CONFIDENCE_THRESHOLD, 0.05,
        help="Minimum confidence for YOLO detections",
    )
    depth_enabled = st.toggle("Depth estimation", value=settings.DEPTH_ENABLED,
                              help="Monocular depth via Depth Anything V2")
    llm_enabled = st.toggle("LLM reasoning", value=settings.LLM_ENABLED,
                            help="GPT-powered scene/risk/decision reasoning")

    if input_mode == "Video":
        sample_rate = st.slider(
            "Sample interval (sec)", 0.5, 5.0,
            settings.SECONDS_PER_SAMPLE, 0.5,
            help="Analyse one frame every N seconds",
        )
        settings.SECONDS_PER_SAMPLE = sample_rate

    settings.CONFIDENCE_THRESHOLD = confidence
    settings.DEPTH_ENABLED = depth_enabled
    settings.LLM_ENABLED = llm_enabled

    st.markdown("---")
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                color: #475569; line-height: 1.6;">
        PIPELINE<br>
        Perception → Scene → Risk → Decision<br><br>
        MODELS<br>
        YOLOv8 · Depth Anything V2 · GPT-4o-mini
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-title">Agentic ADAS Simulator</div>
<div class="main-subtitle">
    Multi-agent driving scene analysis — upload an image or video to begin
</div>
""", unsafe_allow_html=True)


# ── Image Mode ────────────────────────────────────────────────────────────

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

        # Preview
        preview_col, info_col = st.columns([3, 1])
        with preview_col:
            st.image(tmp_path, caption="Input frame", use_container_width=True)
        with info_col:
            st.markdown(f"""
            <div style="padding: 1rem; background: rgba(30,41,59,0.3);
                        border: 1px solid #1E293B; border-radius: 8px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                            color: #64748B; letter-spacing: 0.1em; margin-bottom: 0.6rem;">
                    FILE INFO
                </div>
                <div style="font-family: 'DM Sans', sans-serif; font-size: 0.85rem; color: #CBD5E1;">
                    {uploaded.name}<br>
                    <span style="color: #64748B;">{uploaded.size / 1024:.1f} KB</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Run button
        run_disabled = not backend_ok
        if st.button("Run Analysis", type="primary", use_container_width=True,
                     disabled=run_disabled):
            try:
                with st.spinner("Running ADAS pipeline via backend..."):
                    result = api_analyse_image(tmp_path)
                st.session_state["image_result"] = result
                st.session_state["image_path"] = tmp_path
            except requests.HTTPError as e:
                st.error(f"Backend returned an error: {e}")
            except requests.ConnectionError:
                st.error("Lost connection to backend.")

        # Persistent results
        if "image_result" in st.session_state and "image_path" in st.session_state:
            st.markdown("---")
            show_image_results(
                st.session_state["image_result"],
                st.session_state["image_path"],
            )


# ── Video Mode ────────────────────────────────────────────────────────────

elif input_mode == "Video":
    uploaded = st.file_uploader(
        "Upload a dashcam video",
        type=["mp4", "avi", "mov"],
        help="Short dashcam clip — 10-30 seconds recommended",
    )

    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        st.markdown(f"""
        <div style="text-align: center; font-family: 'JetBrains Mono', monospace;
                    font-size: 0.75rem; color: #64748B; margin: 0.5rem 0 1rem;">
            {uploaded.name} · {uploaded.size / (1024 * 1024):.1f} MB ·
            sampling every {settings.SECONDS_PER_SAMPLE:.1f}s
        </div>
        """, unsafe_allow_html=True)

        run_disabled = not backend_ok
        if st.button("Run Video Analysis", type="primary", use_container_width=True,
                     disabled=run_disabled):
            try:
                with st.spinner("Processing video — this may take a while..."):
                    result = api_analyse_video(tmp_path)
                st.session_state["video_result"] = result
            except requests.HTTPError as e:
                st.error(f"Backend returned an error: {e}")
            except requests.ConnectionError:
                st.error("Lost connection to backend.")

        # Persistent results
        if "video_result" in st.session_state:
            result = st.session_state["video_result"]
            frames = result.get("frames", [])
            output_video_path = result.get("output_video")

            st.markdown("---")

            # ── Summary metrics ───────────────────────────────────────
            decisions = [f.get("decision", {}).get("decision_type", "SAFE")
                         for f in frames]
            worst = max(decisions, key=lambda d: DEC_ORDER.get(d, 0), default="SAFE")
            critical_count = decisions.count("CRITICAL")
            warning_count = decisions.count("WARNING")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Frames Analysed", len(frames))
            m2.metric("Peak Severity", worst)
            m3.metric("Critical Frames", critical_count)
            m4.metric("Warning Frames", warning_count)

            # ── Annotated output video ────────────────────────────────
            if output_video_path:
                video_filename = os.path.basename(output_video_path)
                local_video = api_download_video(video_filename)

                if local_video:
                    st.markdown('<div class="section-header">Annotated Output Video</div>',
                                unsafe_allow_html=True)
                    st.video(local_video)

            # ── Decision timeline ─────────────────────────────────────
            st.markdown('<div class="section-header">Decision Timeline</div>',
                        unsafe_allow_html=True)

            bar_html = '<div style="display:flex;gap:2px;height:32px;margin-bottom:0.6rem;">'
            for f in frames:
                dec = f.get("decision", {}).get("decision_type", "SAFE")
                color = DECISION_COLORS.get(dec, "#64748B")
                ts = f.get("timestamp", 0)
                bar_html += (
                    f'<div style="flex:1;background:{color};opacity:0.7;border-radius:2px;"'
                    f' title="t={ts:.1f}s — {dec}"></div>'
                )
            bar_html += "</div>"

            legend = " ".join(
                f'<span style="margin-right:1rem;font-size:0.75rem;'
                f'font-family:JetBrains Mono,monospace;color:{c};">● {l}</span>'
                for l, c in DECISION_COLORS.items()
            )
            bar_html += f'<div style="text-align:center;">{legend}</div>'
            st.markdown(bar_html, unsafe_allow_html=True)

            # ── Frame detail inspector ────────────────────────────────
            if frames:
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

                selected_idx = st.slider(
                    "Inspect frame", 0, len(frames) - 1, 0,
                    format="Frame %d",
                )

                frame_data = frames[selected_idx]
                ts = frame_data.get("timestamp", 0)
                frame_num = frame_data.get("frame_number", selected_idx)

                st.markdown(f"""
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
                            color: #64748B; margin-bottom: 1rem;">
                    Frame {frame_num} · t = {ts:.1f}s ·
                    Objects detected: {frame_data.get('detected_objects_count', '?')}
                </div>
                """, unsafe_allow_html=True)

                frame_decision = frame_data.get("decision", {})
                frame_risk = frame_data.get("risk_report", {})

                render_decision_banner(frame_decision)

                if frame_risk:
                    render_risk_report(frame_risk)