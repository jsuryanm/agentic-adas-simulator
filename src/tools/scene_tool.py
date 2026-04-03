from typing import List,Tuple 
from loguru import logger
from src.exceptions.custom_exceptions import SceneToolException
from src.models.schemas import SceneSummary


class SceneTool:
    
    VEHICLE_LABELS = {"car",
                      "truck",
                      "bus",
                      "rider",
                      "cyclist"}
    
    PEDESTRIAN_LABELS = {"pedestrian"}

    