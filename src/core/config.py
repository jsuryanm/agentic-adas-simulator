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

    # LaneTool CV 
    LANE_CANNY_LOW: int = 50
    LANE_CANNY_HIGH: int = 150
    LANE_HOUGH_THRESHOLD: int = 30
    LANE_MIN_LINE_LENGTH: int = 40
    LANE_MAX_LINE_GAP: int = 150
    LANE_MIN_SLOPE: float = 0.4
    LANE_MAX_SLOPE: float = 3.0


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