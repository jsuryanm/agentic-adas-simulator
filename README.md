# Agentic ADAS Simulator

A multi-agent simulation of the **decision-making intelligence** inside a modern Advanced Driver Assistance System (ADAS). This project recreates the reasoning pipeline that a real ADAS uses — taking a visual driving scene as input and producing a safety decision as output — using a network of collaborating AI agents orchestrated by LangGraph.

This is **not** a vehicle control system, not real-time, and involves no model training. It simulates the *decision intelligence layer* that sits between raw sensor perception and vehicle actuation — the part that answers: *"Based on what I can see right now, is it safe to continue driving, and if not, what should happen?"*

---

## Table of Contents

- [Why This Architecture](#why-this-architecture)
- [System Architecture](#system-architecture)
- [Pipeline Deep Dive](#pipeline-deep-dive)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Risk & Decision Logic](#risk--decision-logic)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Why This Architecture

A single monolithic neural network that takes an image and outputs a driving decision would be simpler to build — but impossible to debug, test in isolation, or explain to a regulator. Real production ADAS systems don't work that way. They use **specialized subsystems** that each solve one part of the problem and pass structured results to the next stage.

This project mirrors that design using modern agentic AI patterns:

- **Each agent can be developed and tested independently.** You can inject a synthetic scene summary directly into the Risk Agent without running the full perception pipeline.
- **The data flowing between agents is explicit and human-readable.** Every intermediate result is a structured dict that can be inspected, logged, and audited.
- **If a wrong decision is made, you can trace exactly which agent produced the faulty output.** The full reasoning chain — from raw detections through scene understanding, risk scoring, and final decision — is preserved and displayed.
- **Individual agents can be upgraded without touching the rest of the system.** Swap YOLOv8 for a newer model, or replace rule-based risk scoring with a learned model — downstream agents don't change.

This modularity reflects how serious production AI systems are actually built, and makes the system *explainable* — increasingly a regulatory requirement in safety-critical domains.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                              │
│         Dashcam image or video (MP4/AVI/MOV)                │
│         Video → time-based frame sampling (1 fps)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Perception Agent                               │
│  ┌──────────────────┐  ┌────────────────────────┐           │
│  │ YOLOv8s          │  │ Depth Anything V2      │           │
│  │ Object detection │  │ Monocular depth est.   │           │
│  │ + post-filtering │  │ (optional, toggleable) │           │
│  └────────┬─────────┘  └───────────┬────────────┘           │
│           └──────────┬─────────────┘                        │
│    Output: detected_objects[], image_dims                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Scene Understanding Agent                      │
│  ┌──────────────────┐  ┌────────────────────────┐           │
│  │ Scene Tool       │  │ Lane Tool (OpenCV)     │           │
│  │ Rule-based scene │  │ Canny + Hough lines    │           │
│  │ interpretation   │  │ Lane boundary analysis │           │
│  └────────┬─────────┘  └───────────┬────────────┘           │
│           └──────────┬─────────────┘                        │
│  ┌──────────────────────────────────────────┐               │
│  │ LLM Scene Reasoning (optional)          │                │
│  │ GPT-4o-mini for contextual narration    │                │
│  └──────────────────────────────────────────┘               │
│    Output: scene_summary (lead vehicle, pedestrians,        │
│            traffic density, lane status, context notes)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Risk Assessment Agent                          │
│  ┌──────────────────────────────────────────┐               │
│  │ Risk Tool (deterministic, never skipped) │               │
│  │ Scores: collision / pedestrian / lane    │               │
│  │ Composite escalation logic               │               │
│  └──────────────────────────────────────────┘               │
│  ┌──────────────────────────────────────────┐               │
│  │ LLM Risk Validation (optional)           │               │
│  │ Cross-factor interaction check           │               │
│  │ Can escalate risk, never downgrade       │               │
│  └──────────────────────────────────────────┘               │
│    Output: risk_report (individual scores, composite,       │
│            primary driver, explanation)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Decision Agent                                 │
│  ┌──────────────────────────────────────────┐               │
│  │ Policy mapping (deterministic)          │                │
│  │ LOW→SAFE, MED→ADVISORY, HIGH→WARNING,  │                 │
│  │ CRITICAL→CRITICAL                       │                │
│  └──────────────────────────────────────────┘               │
│  ┌──────────────────────────────────────────┐               │
│  │ LLM Decision Reasoning (optional)       │               │
│  │ Situation-specific recommendation text  │               │
│  │ Cannot override the decision type       │               │
│  └──────────────────────────────────────────┘               │
│    Output: decision (type, recommendation,                  │
│            explanation, confidence)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Output Layer                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Streamlit   │  │ FastAPI      │  │ Annotated Video   │  │
│  │ Dashboard   │  │ REST API     │  │ MP4 Output        │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Deep Dive

### 1. Perception Agent

The foundational layer — everything downstream depends on its accuracy. Receives a raw driving image and produces a structured inventory of every detected object.

**What it does:**
- Runs **YOLOv8s** (small model) for object detection with configurable confidence thresholds
- Applies **post-detection filtering** to fix common YOLO mistakes: suppresses false person detections on vehicles, merges overlapping rider/motorcycle pairs, relabels pedestrians on two-wheelers as riders
- Optionally runs **Depth Anything V2** for monocular depth estimation, enriching each detection with a depth-informed distance category (near/mid/far) instead of relying solely on bounding box heuristics
- Computes horizontal position (left/center/right) and distance estimates for each detection

**Writes to state:** `detected_objects`, `image_dims`

### 2. Scene Understanding Agent

Converts raw detections into **driving context**. Knowing "vehicle detected at center-frame, close range" is different from understanding "a vehicle is directly ahead at a distance that may require braking."

**What it does:**
- Identifies the **lead vehicle** (closest center-frame vehicle) and its distance
- Flags **pedestrian proximity** — whether pedestrians are merely visible or actively near the vehicle's projected path
- Estimates **traffic density** (clear/light/moderate/heavy) from detection counts and spatial distribution
- Runs **classical lane detection** (Canny edge detection + Hough line transform) to determine lane status: centered, drifting left, drifting right, or no lane markings detected
- Optionally invokes **GPT-4o-mini** to generate contextual narration and additional scene insights (gracefully falls back to rule-based templates if LLM is disabled or unavailable)

**Writes to state:** `scene_summary`

### 3. Risk Assessment Agent

The analytical core. Scores danger across three independent dimensions, then combines them using composite escalation logic.

**What it does:**
- Applies **deterministic risk rules** (never bypassed, regardless of LLM availability):
  - **Collision risk** — based on lead vehicle presence and proximity
  - **Pedestrian risk** — based on pedestrian detection and path proximity
  - **Lane departure risk** — based on lane status and drift severity
- Computes **composite risk** using escalation logic (see [Risk & Decision Logic](#risk--decision-logic))
- Identifies the **primary risk driver** — which factor contributes most to the composite
- Optionally runs **LLM validation** to check cross-factor interactions the rules can't see (e.g., a near pedestrian combined with lane drift is worse than either alone). The LLM can **escalate** risk levels but **never downgrade** them — safety-conservative by design.

**Writes to state:** `risk_report`

### 4. Decision Agent

Translates risk into actionable driving recommendations.

**What it does:**
- Applies a **fixed policy mapping** from composite risk to decision type (SAFE / ADVISORY / WARNING / CRITICAL)
- Computes a **confidence score** — higher for clear-cut situations (very low or very high risk), lower for ambiguous mid-range scenarios
- Optionally invokes **GPT-4o-mini** to write a situation-specific recommendation and reasoning chain that references the actual objects and hazards in the scene, rather than generic advice. The LLM **cannot override** the decision type — only enriches the explanation.

**Writes to state:** `decision`

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Orchestration | **LangGraph** | Agent pipeline with typed state passing |
| Object Detection | **Ultralytics YOLOv8s** | Pretrained detection, loaded once at init |
| Depth Estimation | **Depth Anything V2 Small** (HuggingFace) | Monocular depth maps, optional |
| Lane Detection | **OpenCV** (Canny + Hough) | Classical computer vision, no ML |
| LLM Reasoning | **GPT-4o-mini** via LangChain | Optional enrichment, never required |
| Dashboard | **Streamlit** | Interactive UI with image/video support |
| API | **FastAPI** | REST endpoints for programmatic access |
| Configuration | **Pydantic Settings** | Typed config with `.env` support |
| Logging | **Loguru** | Structured logging throughout |
| Package Manager | **uv** | Fast Python package management |

---

## Project Structure

```
adas_simulator/
├── pyproject.toml                  ← uv package config (source of truth)
├── .env                            ← environment variables (API keys, toggles)
├── src/
│   ├── core/
│   │   └── config.py               ← all settings via Pydantic Settings
│   ├── models/
│   │   └── schemas.py              ← DetectedObject, SceneSummary, RiskReport, Decision
│   ├── exceptions/
│   │   └── custom_exceptions.py    ← one exception per layer, all inherit ADASBaseException
│   ├── logger/                     ← Loguru configuration
│   ├── tools/
│   │   ├── detection_tool.py       ← YOLO wrapper, model loaded once in __init__
│   │   ├── depth_tool.py           ← Depth Anything V2 wrapper
│   │   ├── lane_tool_cv.py         ← Canny + Hough lane detection
│   │   ├── scene_tool.py           ← Rule-based scene interpretation
│   │   ├── risk_tool.py            ← Risk scoring + composite escalation
│   │   ├── llm_tool.py             ← LangChain chains for scene/risk/decision reasoning
│   │   └── video_tool.py           ← Frame extraction, pipeline orchestration, annotated output
│   ├── agents/
│   │   ├── base_agent.py           ← Abstract base with timing, logging, error handling
│   │   ├── perception_agent.py
│   │   ├── scene_agent.py
│   │   ├── risk_agent.py
│   │   └── decision_agent.py
│   └── pipelines/
│       ├── state.py                ← ADASState TypedDict
│       └── graph.py                ← LangGraph graph construction and execution
├── dashboard/
│   └── streamlit_app.py            ← Streamlit UI (image + video modes)
├── backend/
│   └── app.py                      ← FastAPI REST API
└── tests/                          ← Pytest suite
```

**Dependency direction (strictly enforced):**

```
dashboard / backend → pipeline → agents → tools → models / config
```

No layer imports from a layer above it. Business logic lives in `tools/`; agents only orchestrate.

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- (Optional) OpenAI API key for LLM reasoning features
- (Optional) ffmpeg for browser-compatible annotated video output

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/adas-simulator.git
cd adas-simulator

# Setup package manager 
pip install uv 

# Install all dependencies
uv sync

# Copy environment config
cp .env.example .env
```

Edit `.env` to add your OpenAI API key if you want LLM-enriched reasoning:

```env
OPENAI_API_KEY=sk-...
LLM_ENABLED=true
```

The system works fully without an API key — LLM features degrade gracefully to rule-based fallbacks.

### First Run

```bash
# Start the FastAPI backend
uv run uvicorn backend.app:app --reload 

# In a separate terminal, start the Streamlit dashboard
uv run streamlit run dashboard/streamlit_app.py
```

The dashboard will open in your browser. Upload any driving scene image or short dashcam video to see the pipeline in action.

---

## Usage

### Streamlit Dashboard

The dashboard provides two modes selectable from the sidebar:

**Image mode** — Upload a single driving scene image (JPG/PNG). Click "Run Analysis" to see detected objects with bounding boxes, the scene summary, risk breakdown, and final decision with reasoning chain.

**Video mode** — Upload a dashcam clip (MP4/AVI/MOV, ideally 10–30 seconds). The pipeline samples frames at a configurable interval (default: 1 frame per second), runs the full agent pipeline on each frame, and produces an annotated output video with bounding boxes, risk overlays, and decision labels. A risk timeline chart shows how danger levels evolve across the clip. Note: This can take a while.

**Sidebar controls:**
- Detection confidence threshold (0.1–1.0)
- Depth estimation toggle (on/off)
- LLM reasoning toggle (on/off)
- Video sample interval (0.5–5.0 seconds)

### FastAPI Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Analyse a single image
curl -X POST http://localhost:8000/analyse/image \
  -F "file=@driving_scene.jpg"

# Analyse a video
curl -X POST http://localhost:8000/analyse/video \
  -F "file=@dashcam_clip.mp4"

# Download annotated output video
curl http://localhost:8000/download/video/clip_adas_output.mp4 -o output.mp4
```
---

## Configuration

All configuration is centralised in `src/core/config.py` using Pydantic Settings, with environment variable overrides via `.env`:

| Setting | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolov8s.pt` | Ultralytics model name |
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum detection confidence |
| `DEPTH_ENABLED` | `true` | Enable usage of depth estimation model (depth-anythingv2) |
| `DEPTH_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | HuggingFace model ID |
| `SECONDS_PER_SAMPLE` | `1.0` | Video frame sampling interval |
| `LLM_ENABLED` | `true` | Enable OpenAI LLM reasoning (all three agents) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI LLM model for reasoning chains |
| `LLM_TEMPERATURE` | `0.2` | Low temperature for consistent outputs |
| `LANE_CANNY_LOW/HIGH` | `50/150` | Canny edge detection thresholds |
| `LANE_HOUGH_THRESHOLD` | `30` | Hough line detection sensitivity |

CUDA is auto-detected and used for YOLO and depth models when available; falls back to CPU.

---

## Risk & Decision Logic

### Composite Risk Escalation

Individual risks (collision, pedestrian, lane) are scored independently as LOW / MEDIUM / HIGH / CRITICAL. The composite is computed using escalation rules that account for risk interactions:

| Condition | Composite Result |
|---|---|
| Any individual risk is CRITICAL | **CRITICAL** |
| 2 or more risks are HIGH | **CRITICAL** |
| Exactly 1 risk is HIGH | **HIGH** |
| 2 or more risks are MEDIUM | **HIGH** |
| Otherwise | Maximum of all individual risks |

This escalation logic means two MEDIUM risks together are treated as more dangerous than either alone — reflecting how real-world driving hazards compound.

### Decision Policy

The composite risk maps to a fixed decision type:

| Composite Risk | Decision | Driver Action |
|---|---|---|
| LOW | **SAFE** | Continue normally |
| MEDIUM | **ADVISORY** | Heightened awareness |
| HIGH | **WARNING** | Reduce speed |
| CRITICAL | **CRITICAL** | Brake recommended |

The policy mapping is deterministic and never overridden by the LLM. When LLM reasoning is enabled, it enriches the recommendation text and explanation with scene-specific details, but the decision category itself is always set by rules.

---

## Design Decisions

### Agents 

Agents are orchestrators; tools are executors. The Perception Agent doesn't contain YOLO inference logic — it calls `DetectionTool.detect()`. This means you can swap `DetectionTool` for a different detector (DETR, RT-DETR, a cloud API) without touching the agent. The same separation applies everywhere: `RiskTool` contains the scoring math, `LLMTool` wraps the LangChain chains, and agents only decide *when* and *how* to call them.

### Rule-based risk + optional LLM reasoning 

Safety-critical decisions need deterministic guarantees. The rule-based risk scoring always runs and produces a predictable baseline. The LLM layer is additive — it can catch cross-factor interactions the rules miss (a near pedestrian + lane drift is worse than either alone), but it can only **escalate** risk, never lower it. If the LLM fails, the system still produces a valid decision. This "rules as floor, LLM as ceiling" pattern is common in production safety systems.

### LangGraph state

LangGraph's `TypedDict` state provides a shared blackboard that each agent reads from and writes to. This gives you built-in traceability (every state mutation is visible), testability (inject any state and run a single agent), and the ability to add conditional routing or parallel execution later without refactoring. All values stored in state are plain dicts (schemas call `.to_dict()` / `.model_dump()` before writing) to keep state JSON-serializable.

### Time-based video sampling

Frame-count–based sampling (every Nth frame) produces inconsistent temporal coverage across videos with different framerates. A 30fps clip sampled every 30 frames gives 1 sample/second; a 24fps clip sampled the same way gives 1.25 samples/second. Time-based sampling (`SECONDS_PER_SAMPLE = 1.0`) ensures consistent temporal coverage regardless of the source video's framerate.

---

## Limitations

### Perception Constraints

- **No temporal reasoning.** Each frame is analysed independently with no memory of previous frames. The system cannot track object trajectories, estimate velocities, or detect acceleration/deceleration patterns. A vehicle that has been approaching rapidly for 5 seconds looks the same as one that just appeared.
- **Monocular depth only.** Depth Anything V2 estimates relative depth from a single image. Without stereo vision or LiDAR, absolute distance measurements are not available — "near" and "far" are relative categories, not metric distances.
- **YOLO class limitations.** The pretrained COCO model detects general object categories. It cannot distinguish between a moving pedestrian and a stationary one, doesn't recognize road signs text, and may miss uncommon road users (e-scooters, construction vehicles).

### Scene Understanding Constraints

- **Classical lane detection.** The OpenCV-based Canny + Hough approach works well on clear highway markings but struggles with worn markings, curved roads, intersections, and unmarked roads. It is not a deep learning lane model.
- **No ego-motion estimation.** The system doesn't know whether the camera vehicle is accelerating, braking, or turning. All scene interpretation assumes a static observer.
- **Rule-based scene interpretation.** The scene tool uses threshold-based rules (e.g., "more than 5 vehicles = moderate traffic"). These heuristics are reasonable defaults but not calibrated against real driving datasets.

### Risk & Decision Constraints

- **No time-to-collision.** Without velocity information, risk scoring relies on proximity alone. A vehicle 5 metres ahead at a standstill and one 5 metres ahead approaching at 60 km/h receive similar risk scores.
- **Fixed decision policy.** The risk-to-decision mapping is a static lookup table, not a learned or adaptive policy. It doesn't account for road type (highway vs. school zone), speed limits, or jurisdictional regulations.
- **LLM latency.** When enabled, the three LLM calls (scene reasoning, risk validation, decision explanation) add 2–5 seconds per frame. This is acceptable for offline analysis but precludes real-time operation.

### System-Level Constraints

- **Not real-time.** Pipeline latency (YOLO + optional depth + lane detection + optional LLM) is in the range of 1–8 seconds per frame depending on configuration and hardware. Real ADAS operates at 10–30 Hz.
- **Single camera only.** Real ADAS fuses data from multiple cameras, radar, LiDAR, and ultrasonic sensors. This system processes one front-facing camera view.
- **No vehicle control interface.** Decisions are informational only — the system outputs recommendations but has no mechanism to actuate braking, steering, or throttle.

---

## Future Improvements


- **Multi-frame tracking.** Integrate a tracker (ByteTrack, BoT-SORT) to maintain object identities across frames, enabling velocity estimation, trajectory prediction, and time-to-collision calculations.
- **Deep learning lane detection.** Replace the classical Hough-based approach with a model like CLRNet or LaneATT for robust performance on curved roads, worn markings, and complex intersections.
- **Weather and lighting classification.** Add a lightweight classifier to detect rain, fog, night, and glare conditions, then adjust detection confidence thresholds and risk weights accordingly.
- **Scenario library.** Build a curated set of test scenarios (highway cruising, urban intersection, pedestrian crossing, construction zone, night driving) with expected outputs for regression testing.

### Long-Term Vision

- **Hardware-in-the-loop testing.** Connect the pipeline to a driving simulator (CARLA, LGSVL) to process live camera feeds and validate decisions against ground-truth scenarios.
- **Edge deployment.** Optimise the pipeline for edge hardware (Jetson, Coral) by quantizing YOLO and depth models, removing LLM dependency, and targeting sub-100ms latency per frame.
- **Feedback learning.** Implement a feedback loop where human reviewers can flag incorrect decisions, building a dataset for supervised fine-tuning of the risk scoring model.
- **Multi-agent negotiation.** Extend the linear pipeline to allow agents to query each other — for example, the Risk Agent requesting the Perception Agent to re-examine a specific region of the image at higher resolution when confidence is low.

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## License

This project is built for educational and portfolio purposes. See `LICENSE` for details.