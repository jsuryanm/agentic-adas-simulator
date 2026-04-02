from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field 
import torch

class Config(BaseSettings):
    YOLO_MODEL: str = Field(default="yolov8s.pt",
                            description="YOLO model name")
    YOLO_DEVICE: str = Field(default="cuda" if torch.cuda.is_available else "cpu")
    
    CONFIDENCE_THRESHOLD: float = Field(default=0.35,
                                        ge=0,
                                        le=1)
    
    NEAR_DISTANCE_THRESHOLD: float = 0.35
    MID_DISTANCE_THRESHOLD: float = 0.15

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