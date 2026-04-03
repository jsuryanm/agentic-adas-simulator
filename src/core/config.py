from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field 
import torch

class Config(BaseSettings):
    YOLO_MODEL: str = Field(default="yolov8s.pt",
                            description="YOLO model name")
    YOLO_DEVICE: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    
    CONFIDENCE_THRESHOLD: float = Field(default=0.35,
                                        ge=0,
                                        le=1)
    
    NEAR_DISTANCE_THRESHOLD: float = 0.35
    MID_DISTANCE_THRESHOLD: float = 0.15

    # Depth Estimation (DepthAnything V2)
    DEPTH_ENABLED: bool = True
    DEPTH_MODEL: str = "depth-anything/Depth-Anything-V2-Small-hf"
    DEPTH_DEVICE: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    DEPTH_NEAR_THRESHOLD: float = 0.6
    DEPTH_MID_THRESHOLD: float = 0.3

    # Lane Detection 
    LANE_ENABLED: bool = True 
    LANE_MODEL_PATH: str = "models/ufld_v2_culane_res18.onnx"

    # Model architecture constants (CULane ResNet-18 defaults)
    LANE_INPUT_HEIGHT: int = 320
    LANE_INPUT_WIDTH: int = 1600
    LANE_NUM_LANES: int = 4
    LANE_NUM_ROW_ANCHORS: int = 18
    LANE_NUM_GRIDDING: int = 200
    LANE_ROW_ANCHOR_START: int = 121
    LANE_ROW_ANCHOR_END: int = 301
 
    # Confidence thresholds
    LANE_EXISTENCE_THRESHOLD: float = 0.5
    LANE_POINT_THRESHOLD: float = 0.5
    LANE_MIN_POINTS: int = 4


    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "adas.log"

    # VIDEO SAMPLING 
    SECONDS_PER_SAMPLE: float = 1.0

    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = Field(default=0.2,ge=0,le=1)
    LLM_MAX_TOKENS: int = 400
    LLM_ENABLED: bool = True 
    

    model_config = SettingsConfigDict(env_file=".env",
                                      env_file_encoding="utf-8",
                                      case_sensitive=False) 

settings = Config()