import os 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]: %(message)s")


list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/agents/__init__.py",
    "src/agents/base_agent.py",
    "src/agents/decision_agent.py",
    "src/agents/perception_agent.py",
    "src/agents/risk_agent.py",
    "src/agents/scene_agent.py",
    "src/models/__init__.py",
    "src/models/schemas.py",
    "src/pipelines/__init__.py",
    "src/pipelines/graph.py",
    "src/pipelines/state.py",
    "src/tools/__init__.py",
    "src/tools/detection_tool.py",
    "src/tools/llm_tool.py",
    "src/tools/risk_tool.py",
    "src/tools/scene_tool.py",
    "src/tools/depth_tool.py",
    "src/tools/lane_tool.py",
    "src/exceptions/__init__.py",
    "src/exceptions/custom_exceptions.py",
    "src/logger/__init__.py",
    "src/logger/custom_logger.py",
    "src/mcp_server/__init__.py",
    "src/mcp_server/server.py",
    "src/core/__init__.py",
    "src/core/config.py",
    "backend/__init__.py",
    "backend/app.py",
    "dashboard/__init__.py",
    "dashboard/app.py",
    "tests/__init__.py",
    "tests/test_detection.py",
    "requirements.txt",
    ".env",]


for file_path in list_of_files:
    file_path =  Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating an empty file: {file_path}")
    
    else:
        logging.info(f"{file_name} already exists")
