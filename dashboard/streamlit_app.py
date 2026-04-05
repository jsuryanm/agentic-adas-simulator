import os
import tempfile
import requests
import cv2
import streamlit as st

from src.core.config import settings
from src.tools.video_tool import annotate_image

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic ADAS Simulator",
                   layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.title("ADAS Simulator")

input_mode = st.sidebar.radio("Input mode", ["Image", "Video"])
st.sidebar.markdown("Settings")

confidence = st.sidebar.slider("Detection confidence", 0.1, 1.0, settings.CONFIDENCE_THRESHOLD, 0.05)
depth_enabled = st.sidebar.checkbox("Enable depth estimation", settings.DEPTH_ENABLED)
llm_enabled = st.sidebar.checkbox("LLM reasoning", settings.LLM_ENABLED)
sample_rate = st.sidebar.slider("Video sampling rate", 0.5, 5.0, settings.SECONDS_PER_SAMPLE, 0.5)

settings.CONFIDENCE_THRESHOLD = confidence
settings.DEPTH_ENABLED = depth_enabled
settings.LLM_ENABLED = llm_enabled
settings.SECONDS_PER_SAMPLE = sample_rate


# ── Results display ───────────────────────────────────────────────────────

def show_results(results: dict, image_path=None):
    """Render pipeline results in the main area."""
    decision = results.get("decision", {})

    st.subheader("Decision")
    dec_type = decision.get("decision_type", "SAFE")
    st.text(f"Type: {dec_type}")
    st.text(f"Recommendation: {decision.get('recommendation', '')}")
    st.text(f"Confidence: {decision.get('confidence', '')}")

    if image_path:
        try:
            annotated = annotate_image(image_path, results)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated,
                     caption="ADAS Annotated Output",
                     use_container_width=True)
        except Exception as e:
            st.warning(str(e))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Detected Objects")
        objects = results.get("detected_objects", [])
        if objects:
            for obj in objects:
                label = obj.get("label", "?")
                conf = obj.get("confidence", 0)
                pos = obj.get("position", "?")
                dist = obj.get("distance", "?")
                st.text(f"{label}  {conf:.2f}  {pos}  {dist}")
        else:
            st.text("No objects")

    with col2:
        st.subheader("Scene Summary")
        scene = results.get("scene_summary", {})
        if scene:
            st.text(f"Lead Vehicle: {scene.get('lead_vehicle_present')}")
            st.text(f"Distance: {scene.get('lead_vehicle_distance')}")
            st.text(f"Pedestrian: {scene.get('pedestrian_present')}")
            st.text(f"Traffic: {scene.get('traffic_density')}")
            st.text(f"Lane: {scene.get('lane_status')}")
        else:
            st.text("No scene data")

    with col3:
        st.subheader("Risk Report")
        risk = results.get("risk_report", {})
        if risk:
            st.text(f"Collision: {risk.get('collision_risk')}")
            st.text(f"Pedestrian: {risk.get('pedestrian_risk')}")
            st.text(f"Lane: {risk.get('lane_risk')}")
            st.text(f"Composite: {risk.get('composite_risk')}")
        else:
            st.text("No risk data")


# ── Main content area ─────────────────────────────────────────────────────

if input_mode == "Image":
    st.title("Agentic ADAS Image Analysis")

    uploaded = st.file_uploader("Upload Image",
                                type=["jpg", "png", "jpeg"])

    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.image(tmp_path, caption="Uploaded image")

        if st.button("Run Analysis"):
            with st.spinner("Processing..."):
                with open(tmp_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(
                        url=f"{API_URL}/analyse/image",
                        files=files,
                    )

                result = response.json()

            show_results(result, tmp_path)

elif input_mode == "Video":
    st.title("Agentic ADAS Video Analysis")

    uploaded = st.file_uploader("Upload video",
                                type=["mp4", "avi", "mov"])

    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("Run Video Analysis"):
            with st.spinner("Processing video..."):
                with open(tmp_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(
                        url=f"{API_URL}/analyse/video",
                        files=files,
                    )

                result = response.json()

            st.success("Processing Complete")

            st.subheader("Analysis Summary")
            st.text(f"Frames analysed: {result.get('total_frames_analysed')}")

            if result.get("output_video"):
                video_name = os.path.basename(result["output_video"])

                video_url = f"{API_URL}/download/video/{video_name}"
                video_bytes = requests.get(video_url).content
                st.subheader("ADAS Output Video")
                
                st.video(video_bytes)

            for frame in result.get("frames", []):
                dec_type = frame["decision"].get("decision_type", "?")
                st.text(
                    f"Frame {frame['frame_number']} — "
                    f"Decision: {dec_type}"
                )