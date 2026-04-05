# Agentic ADAS Simulator

A multi-agent simulation of the decision-making intelligence inside a modern Advanced Driver Assistance System (ADAS). Processes driving images and videos through a 4-agent LangGraph pipeline and displays results on a Streamlit dashboard.

## Architecture

```
[Driving Scene Image/Video]
        ↓
[Perception Agent]    → YOLO detection + Depth estimation
        ↓
[Scene Agent]         → Semantic scene summary + Lane detection
        ↓
[Risk Agent]          → Collision / Pedestrian / Lane risk scoring
        ↓
[Decision Agent]      → SAFE / ADVISORY / WARNING / CRITICAL 
        ↓
[Dashboard + Annotated Video Output]
```

## Tech Stack

- **Orchestration**: LangGraph (agent pipeline)
- **Perception**: YOLOv8 (object detection), Depth Anything V2 (depth estimation), OpenCV (lane detection)
- **Reasoning**: Rule-based scoring + optional LLM validation (GPT-4o-mini)
- **UI**: Streamlit dashboard with video support
- **API**: FastAPI backend

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Copy and edit environment config
cp .env.example .env
# Add your OPENAI_API_KEY if you want LLM reasoning

# 3. Run the dashboard
uv run streamlit run dashboard/app.py

# 4. (Optional) Run the API server
uv run uvicorn backend.app:app --reload


# 5. Run tests
uv run pytest tests/ -v
```

## Features

- **Image analysis**: Upload a driving scene image for instant ADAS analysis
- **Video analysis**: Upload a dashcam clip (10-30s) for frame-by-frame analysis
- **Annotated video output**: Download an MP4 with bounding boxes, risk levels, and decisions overlaid
- **Risk timeline**: See how risk changes across video frames
- **LLM enrichment**: Optional GPT-4o-mini for contextual reasoning (falls back to rules if disabled)

## Project Structure

```
adas_simulator/
├── pyproject.toml              ← uv package config (source of truth)
├── .env                        ← environment variables
├── src/
│   ├── core/config.py          ← all settings (Pydantic)
│   ├── models/schemas.py       ← DetectedObject, SceneSummary, RiskReport, Decision
│   ├── tools/
│   │   ├── detection_tool.py   ← YOLO wrapper
│   │   ├── depth_tool.py       ← Depth Anything V2 wrapper
│   │   ├── lane_tool_cv.py     ← Classical CV lane detection
│   │   ├── scene_tool.py       ← Rule-based scene analysis
│   │   ├── risk_tool.py        ← Risk scoring + composite logic
│   │   ├── llm_tool.py         ← LLM reasoning chains
│   │   └── video_tool.py       ← Frame extraction + annotated video output
│   ├── agents/
│   │   ├── base_agent.py       ← Abstract base with timing/logging
│   │   ├── perception_agent.py
│   │   ├── scene_agent.py
│   │   ├── risk_agent.py
│   │   └── decision_agent.py
│   ├── pipelines/
│   │   ├── state.py            ← ADASState TypedDict
│   │   └── graph.py            ← LangGraph pipeline
│   └── logger/                 ← Loguru setup
├── dashboard/app.py            ← Streamlit UI
├── backend/app.py              ← FastAPI REST API
└── tests/                      ← Pytest suite
```

## Risk Composite Logic

| Condition | Composite |
|---|---|
| Any CRITICAL | CRITICAL |
| 2+ HIGH | CRITICAL |
| 1 HIGH | HIGH |
| 2+ MEDIUM | HIGH |
| Otherwise | Max individual |

## Decision Policy

| Composite Risk | Decision | Action |
|---|---|---|
| LOW | SAFE | Continue normally |
| MEDIUM | ADVISORY | Heightened awareness |
| HIGH | WARNING | Reduce speed |
| CRITICAL | CRITICAL | Brake recommended |
