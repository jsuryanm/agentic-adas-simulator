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


st.sidebar.title("ADAS Simulator")

input_mode = st.sidebar.radio("Input mode",["Image","Video"])
st.sidebar.markdown("Settings")

confidence = st.sidebar.slider("Detection confidence",0.1,1.0,settings.CONFIDENCE_THRESHOLD,0.05)
depth_enabled = st.sidebar.checkbox("Enable depth estimation",settings.DEPTH_ENABLED)

llm_enabled = st.sidebar.checkbox("LLM reasoning",settings.LLM_ENABLED)

sample_rate = st.sidebar.slider("Video sampling rate",0.5,5.0,settings.SECONDS_PER_SAMPLE,0.5)

settings.CONFIDENCE_THRESHOLD = confidence
settings.DEPTH_ENABLED = depth_enabled
settings.LLM_ENABLED = llm_enabled
settings.SECONDS_PER_SAMPLE = sample_rate


def show_results(results: dict,image_path = None):
    decision = results.get("decision",{})

    st.subheader("Decision")

    st.text(f"Type: {decision.get("decision_type","SAFE")}")
    st.text(f"Recommendations: {decision.get("recommendation","")}")
    st.text(f"Confidence: {decision.get("confidence","")}")

    if image_path:
        try:
            annotated = annotate_image(image_path,results)
            annotated = cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)
            st.image(annotated,
                     caption="Annotated frame",
                     use_container_width=True)
        
        except Exception as e:
            st.warning(str(e))