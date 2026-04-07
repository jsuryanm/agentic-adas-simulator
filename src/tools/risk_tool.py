from src.models.schemas import RiskReport,RiskLevel,LaneStatus
from src.exceptions.custom_exceptions import RiskToolException

from src.logger.custom_logger import logger 
from pprint import pprint


class RiskTool:
    
    def assess(self,scene_summary: dict) -> RiskReport:
        """Produce a RiskReport from a SceneSummary dict.

        Args:
            scene_summary: SceneSummary.model_dump() dict.

        Returns:
            RiskReport pydantic model.
        """
        try:
            collision = self._score_collision(scene_summary)
            pedestrian = self._score_pedestrian(scene_summary)
            lane = self._score_lane(scene_summary)

            composite = self._composite(collision,pedestrian,lane)
            primary = self._primary_driver(collision,pedestrian,lane)
            explanation = self._explain(collision,pedestrian,lane,composite,scene_summary)

            report = RiskReport(collision_risk=collision,
                                pedestrian_risk=pedestrian,
                                lane_risk=lane,
                                composite_risk=composite,
                                primary_driver=primary,
                                explanation=explanation)
            

            logger.info(
                f"Risk assessed: collision={collision.value}, "
                f"pedestrian={pedestrian.value}, lane={lane.value}, "
                f"composite={composite.value}"
            )
            return report
        
        except Exception as e:
            raise RiskToolException(f"Risk assessment failed: {e}")

    def _score_collision(self,summary: dict) -> RiskLevel:
        if not summary.get("lead_vehicle_present", False):
            return RiskLevel.LOW 
        
        distance = summary.get("lead_vehicle_distance","far")
        density = summary.get("traffic_density","clear")

        if distance == "near":
            return RiskLevel.CRITICAL if density in ("moderate","congested") else  RiskLevel.MEDIUM
        
        elif distance == "mid":
            return RiskLevel.HIGH if density == "congested" else RiskLevel.MEDIUM
        
        return RiskLevel.LOW 
    
    def _score_pedestrian(self,summary: dict) -> RiskLevel:
        if not summary.get("pedestrian_present",False):
            return RiskLevel.LOW 
        
        near_path = summary.get("pedestrian_near_path",False)

        if near_path:
            return RiskLevel.CRITICAL
        return RiskLevel.MEDIUM
    
    def _score_lane(self,summary: dict) -> RiskLevel:
        """Measures the vehicle's risk level based on whether vehicle is centered or drifting"""
        status = summary.get("lane_status","centered")
        if status in (LaneStatus.DRIFTING_LEFT.value, LaneStatus.DRIFTING_LEFT):
            return RiskLevel.MEDIUM
        if status in (LaneStatus.DRIFTING_RIGHT.value, LaneStatus.DRIFTING_RIGHT):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _composite(self,
                   collision: RiskLevel,
                   pedestrian: RiskLevel,
                   lane: RiskLevel) -> RiskLevel:
        """Combines 3 risks collision,pedestrian and lane risks into one final risk"""

        levels = [collision,pedestrian,lane]
        scores = [r.score() for r in levels]

        if RiskLevel.CRITICAL in levels:
            return RiskLevel.CRITICAL
        
        high_count = sum(r for r in levels if r == RiskLevel.HIGH)

        if high_count >= 2:
            return RiskLevel.CRITICAL
        
        if high_count == 1:
            return RiskLevel.HIGH
        
        med_count = sum(1 for r in levels if r == RiskLevel.MEDIUM)
        if med_count >= 2:
            return RiskLevel.HIGH
        
        max_score = max(scores)
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH,
                      RiskLevel.MEDIUM, RiskLevel.LOW]:
            if level.score() == max_score:
                return level
        return RiskLevel.LOW
    

    def _primary_driver(self,
                        collision: RiskLevel,
                        pedestrian: RiskLevel,
                        lane: RiskLevel) -> str:
        """Finds which risk contributed most to the final decision"""
        
        ranked = [("collision_risk",collision),
                  ("pedestrian_risk",pedestrian),
                  ("lane_risk",lane)]
        
        ranked.sort(key=lambda x:x[1].score(),reverse=True)
        return ranked[0][0]
    
    def _explain(self,
                 collision: RiskLevel,
                 pedestrian: RiskLevel,
                 lane: RiskLevel,
                 composite: RiskLevel,
                 summary: dict) -> dict:

        parts = []

        if collision.score() >= RiskLevel.MEDIUM.score():
            dist = summary.get("lead_vehicle_distance","unknown")
            parts.append(f"Lead vehicle at {dist} range poses {collision.value} collision risk.")
        
        if pedestrian.score() >= RiskLevel.MEDIUM.score():
            near = "in path" if summary.get("pedestrian_near_path") else "nearby"
            parts.append(f"Pedestrian {near} poses {pedestrian.value} risk.")
        
        if lane.score() >= RiskLevel.MEDIUM.score():
            status = summary.get("lane_status", "unknown")
            parts.append(f"Lane status '{status}' raises {lane.value} drift risk.")

        if not parts:
            parts.append("No significant risks detected at this time.")

        parts.append(f"Composite risk level: {composite.value}.")
        return " ".join(parts)

# if __name__ == "__main__":
#     risk_tool = RiskTool()

#     # Simulated SceneSummary output
#     scene_summary = {
#         "lead_vehicle_present":True,
#         "lead_vehicle_distance":"near",
#         "pedestrian_present":True,
#         "pedestrian_near_path":True,
#         "traffic_density":"moderate",
#         "lane_status":"drifting_right",
#         "context_notes":[
#             "Lead car very close",
#             "Pedestrian in path"
#         ]
#     }

#     report = risk_tool.assess(scene_summary)

#     print("\n===== RISK REPORT =====")

#     pprint(report.model_dump())