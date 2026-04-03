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

    def __init__(self):
        self.logger = logger.bind(tool="SceneTool")

    def analyze(self,detected_objects: List[dict]) -> SceneSummary:
        """
        Build a SceneSummary from detection dicts

        Args:
            detected_objects: List of DetectedObject.to_dict() results.

        Returns:
            SceneSummary with semantic driving context.
        """
        try:
            self.logger.info(f"Analysing scene with {len(detected_objects)} detections")

            notes: List[str] = []

            lead_vehicle_present,lead_vehicle_distance = self._check_lead_vehicles(detected_objects,notes)
            pedestrian_present,pedestrian_near_path = self._check_pedestrians(detected_objects,notes)
            traffic_density = self._estimate_traffic_density(detected_objects,notes)
            lane_status = self._estimate_lane_status(detected_objects,notes)

            summary = SceneSummary(lead_vehicle_present=lead_vehicle_present,
                                   lead_vehicle_distance=lead_vehicle_distance,
                                   pedestrian_present=pedestrian_present,
                                   pedestrian_near_path=pedestrian_near_path,
                                   traffic_density=traffic_density,
                                   lane_status=lane_status,
                                   context_notes=notes)
            
            self.logger.info(f"Scene summary built | Lead vehicle distance: {lead_vehicle_distance} | Pedestrian present: {pedestrian_present} | Traffic density: {traffic_density}")
            return summary

        except Exception as e:
            raise SceneToolException(f"Scene analysis failed: {e}")
        
    def _check_lead_vehicle(self,
                            objects: List[dict],
                            notes: List[str]) -> Tuple[bool,str]:
        """Find the closest center vehicle"""
        