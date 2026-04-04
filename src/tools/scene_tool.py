from typing import List,Optional 
from loguru import logger 
from pprint import pprint

from src.models.schemas import SceneSummary,LaneAnalysis,LaneStatus
from src.exceptions.custom_exceptions import SceneToolException

class SceneTool:

    VEHICLE_LABELS = {"car","truck","bus","motorcycle","rider","cyclist"}
    PEDESTRIAN_LABELS = {"pedestrian","person"}

    def analyse(self,
                detected_objects: List[dict],
                lane_analysis: Optional[dict] = None) -> SceneSummary:
        """Build a SceneSummary from detections + lane data.

        Args:
            detected_objects: list of DetectedObject dicts.
            lane_analysis: LaneAnalysis dict (may be None if lane
                           detection was skipped or failed).
        Returns:
            SceneSummary pydantic model.
        """

        try:
            vehicles =  [obj for obj in detected_objects 
                        if obj.get("label") in self.VEHICLE_LABELS]
        
            pedestrians = [obj for obj in detected_objects 
                           if obj.get("label") in self.PEDESTRIAN_LABELS] 
        
            lead = self._find_lead_vehicle(vehicles)
            lead_vehicle_present = lead is not None
            lead_vehicle_distance = lead.get("distance","far") if lead else "far"

            pedestrian_present = len(pedestrians) > 0 
            
            pedestrian_near_path = any(p.get("position") == "center" 
                                       and p.get("distance") in ("near", "mid")
                                       for p in pedestrians)
            
            traffic_density = self._classify_traffic(vehicles)
            lane_status = self._derive_lane_status(lane_analysis)
            notes = self._build_context_notes(vehicles,
                                              pedestrians,
                                              lead,
                                              lane_analysis)
            
            summary = SceneSummary(lead_vehicle_present=lead_vehicle_present,
                                   lead_vehicle_distance=lead_vehicle_distance,
                                   pedestrian_present=pedestrian_present,
                                   pedestrian_near_path=pedestrian_near_path,
                                   traffic_density=traffic_density,
                                   lane_status=lane_status,
                                   context_notes=notes)
            
            logger.info(f"Scene summary built: traffic = {traffic_density}, lead:{lead_vehicle_distance}, lane: {lane_status.value}")
            return summary
        except Exception as e:
            raise SceneToolException(f"Scene analysis failed: {e}")
        
    
    def _find_lead_vehicle(self,vehicles: List[dict]) -> Optional[dict]:
        """Find the closest vehicle in center lane"""
        
        center_vehicles = [v for v in vehicles if v.get("position") == "center"]

        if not center_vehicles:
            return None 
        
        distance_priority = {"near": 0,"mid": 1,"far": 2}

        center_vehicles.sort(key=lambda v:distance_priority.get(v.get("distance","far"),3))

        return center_vehicles[0]
    
    def _classify_traffic(self,vehicles: List[dict]) -> str:
        """Classify traffic density based on vehicle count"""
        count = len(vehicles)

        if count == 0:
            return "clear"
        
        elif count <= 3:
            return "light"
        
        elif count <= 7:
            return "moderate"
        return "congested"
    
    def _derive_lane_status(self,lane_analysis: Optional[dict]) -> LaneStatus:
        """Convert lateral lane offset into lane status enum"""
        
        if lane_analysis is None:
            return LaneStatus.CENTERED
        
        offset = lane_analysis.get("lateral_offset",0.0)

        if offset < -0.3:
            return LaneStatus.DRIFTING_LEFT
        
        elif offset > 0.3:
            return LaneStatus.DRIFTING_RIGHT
        
        return LaneStatus.CENTERED
    
    def _build_context_notes(self,
                             vehicles: List[dict],
                             pedestrians: List[dict],
                             lead: Optional[dict],
                             lane_analysis: Optional[dict]) -> List[str]:
        """Generate human-readable context notes for the scene."""

        notes: List[str] = [] 

        # lead vehicle warning 
        if lead and lead.get("distance") == "near":
            notes.append(f"Lead {lead.get("label","vehicle")} is close at range (confidence: {lead.get("confidence",0):.0%})")


        # pedestrian alert
        crossing = [p for p in pedestrians 
                    if p.get("position") == "center" 
                    and p.get("distance") == "near"]
        
        if crossing:
            notes.append(f"{len(crossing)} pedestrian(s) in path at close range")
        
        riders = [v for v in vehicles 
                  if v.get("label") in ("rider", "cyclist")]
        
        if riders:
            notes.append(f"{len(riders)} two-wheeler(s) detected nearby")

        # Lane info
        if lane_analysis:
            if not lane_analysis.get("road_detected", True):
                notes.append("Lane markings not detected — road may be unmarked")
            elif lane_analysis.get("road_coverage", 1.0) < 0.5:
                notes.append("Only partial lane markings visible")

        if not notes:
            notes.append("No unusual conditions detected")

        return notes
    
# if __name__ == "__main__":

#     scene_tool = SceneTool()

#     # Simulated detections (normally from DetectionTool)
#     detected_objects = [
#         {
#             "label":"car",
#             "confidence":0.91,
#             "position":"center",
#             "distance":"near",
#             "bbox":[100,200,300,400]
#         },
#         {
#             "label":"pedestrian",
#             "confidence":0.88,
#             "position":"center",
#             "distance":"mid",
#             "bbox":[500,220,580,420]
#         },
#         {
#             "label":"truck",
#             "confidence":0.85,
#             "position":"left",
#             "distance":"far",
#             "bbox":[50,100,200,300]
#         }
#     ]

#     # Simulated lane analysis (normally from LaneTool)
#     lane_analysis = {
#         "lateral_offset":0.42,
#         "lane_width_px":820,
#         "road_coverage":1.0,
#         "road_detected":True
#     }

#     summary = scene_tool.analyse(
#         detected_objects,
#         lane_analysis
#     )

#     print("\n===== SCENE SUMMARY =====")

#     pprint(summary.model_dump())