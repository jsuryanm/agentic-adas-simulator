import os 
import sys 
import tempfile 
import shutil 

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from src.pipelines.graph import run_pipeline
from src.tools.video_tool import process_video 
from src.core.config import settings 

app = FastAPI(title="Agentic ADAS Simulator API",
              description="Agentic ADAS pipeline, upload any image or video for analysis",
              version="1.0.0")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/analyse/image")
async def analyze_image(file: UploadFile = File(...)):
    """Run ADAS pipeline on single image"""
    suffix = os.path.splitext(file.filename or "image.jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        state = run_pipeline(tmp_path)
        return {
            "detected_objects": state.get("detected_objects", []),
            "scene_summary": state.get("scene_summary"),
            "risk_report": state.get("risk_report"),
            "decision": state.get("decision"),
            "processing_time": state.get("processing_time", {}),
            "errors": state.get("errors", []),
        }
    finally:
        os.unlink(tmp_path)

@app.post("/analyse/video")
async def analyse_video(file: UploadFile = File(...)):
    """Run the ADAS pipeline on a video and return frame results + output path."""
    suffix = os.path.splitext(file.filename or "video.mp4")[1]
    
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
     
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name 

    try:
        results = process_video(tmp_path)

        frames_summary = []
        for f in results.get("frames",[]):
            s = f["state"]
            frames_summary.append({"frame_number": f["frame_number"],
                                    "timestamp": f["timestamp"],
                                    "decision":s.get("decision",{}),
                                    "risk_report":s.get("risk_report",{}),
                                    "detected_objects_count":len(s.get("detected_objects",[]))})
            
        return {"frames":frames_summary,
                "output_video":results.get("output_video"),
                "total_frames_analysed":len(frames_summary)}
    
    finally:
        os.unlink(tmp_path)

@app.get("/download/video/{filename}")
def download_video(filename: str):
    """Download a generated output video."""
    path = os.path.join(settings.VIDEO_OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)