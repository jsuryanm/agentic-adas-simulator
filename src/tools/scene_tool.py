from typing import Optional,List
from src.models.schemas import SceneSummary,LaneStatus
from src.core.config import settings 

from src.exceptions.custom_exceptions import SceneToolException
from loguru import logger 

VEHICLE_LABELS = {"car", "truck", "bus", "rider", "cyclist"}

PEDESTRIAN_LABELS = {"pedestrian"}

class SceneTool:
    """Translates raw perception outputs into a semantic SceneSummary"""

    def analyze(self,
                detected_objects: List[dict],
                lane_analysis: Optional[dict],
                img_dims: Optional[dict] = None) -> SceneSummary:
        """
        Build a SceneSummary from perception data.

        Args:
            detected_objects: List of DetectedObject dicts
                              (label, confidence, position, distance, bbox, ...).
            lane_analysis:    LaneAnalysis dict
                              (lateral_offset, lane_width_px, road_coverage, road_detected).
            img_dims:         {"width": int, "height": int}  (optional, for future use).

        Returns:
            SceneSummary (Pydantic model).
        """

        pass 

    def _find_lead_vehicle(self,objects: List[dict]) -> Optional[dict]:
        """
        Find the most relevant vehicle ahead (center position, closest).

        Priority: center vehicles first, then pick the nearest one.
        """

        centered_vehicles = [obj for obj in objects 
                             if obj.get("label") in VEHICLE_LABELS
                             and obj.get("position") == "center"]
        
        if not centered_vehicles:
            return None 
        
        distance_order = {"near":0,"mid":1,"far":2}
        centered_vehicles.key()