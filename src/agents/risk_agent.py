from src.agents.base_agent import BaseAgent
from src.tools.risk_tool import RiskTool
from src.tools.llm_tool import LLMTool
from src.pipelines.state import ADASState

class RiskAgent(BaseAgent):
    name = "risk_agent"

    def __init__(self):
        super().__init__()
        self.risk_tool = RiskTool()
        self.llm_tool = LLMTool()

    def _process(self,state: ADASState) -> dict:
        scene_summary = state.get("scene_summary")

        if scene_summary is None:
            self.logger.warning("No scene_summary available - defaulting to low risk")
            return {
                "risk_report": {
                    "collision_risk": "low",
                    "pedestrian_risk": "low",
                    "lane_risk": "low",
                    "composite_risk": "low",
                    "primary_driver": "none",
                    "explanation": "No scene data available for risk assessment.",
                },
            }
        
        # deterministic rules (never bypassed)
        report = self.risk_tool.assess(scene_summary)
        report_dict = report.model_dump()

        self.logger.info(
            f"Rule-based risk: composite={report_dict['composite_risk']}, "
            f"driver={report_dict['primary_driver']}"
        )

        # LLM validates cross factor interactions and may escalate
        detected_objects = state.get("detected_objects",[])
        try:
            report_dict = self.llm_tool.validate_risk(report_dict,scene_summary,detected_objects)
            self.logger.info(f"Final risk after llm validation: composite={report_dict['composite_risk']}")
        
        except Exception as e:
            self.logger.warning(f"LLM risk validation failed, keeping rules: {e}")
        
        return {"risk_report":report_dict}
