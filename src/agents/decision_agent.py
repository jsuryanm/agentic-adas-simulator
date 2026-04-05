from src.agents.base_agent import BaseAgent
from src.tools.llm_tool import LLMTool
from src.pipelines.state import ADASState

DECISION_POLICY = {"low":"SAFE",
                   "medium":"ADVISORY",
                   "high":"WARNING",
                   "critical":"CRITICAL"}

DEFAULT_RECOMMENDATIONS = {"SAFE":"Continue current driving behaviour. No hazards detected.",
                           "ADVISORY":"Heightened awareness recommended. Monitor surroundings closely.",
                           "WARNING": "Reduce speed and increase following distance.",
                           "CRITICAL": "Brake recommended immediately. Collision risk is high."}

class DecisionAgent(BaseAgent):
    """Produces a final driving recommendation from the risk report.
    Reads:  risk_report, scene_summary, detected_objects
    Writes: decision
    """
    name = "decision_agent"

    def __init__(self):
        super().__init__()
        self._llm_tool = LLMTool()
    
    def _process(self,state: ADASState) -> dict:
        risk_report = state.get("risk_report")

        if risk_report is None:
            self.logger.warning("No risk report defaulting to SAFE")
            return {
                "decision": {
                    "decision_type": "SAFE",
                    "recommendation": DEFAULT_RECOMMENDATIONS["SAFE"],
                    "explanation": "Risk report unavailable; assuming safe.",
                    "confidence": 0.5,
                }
            }
        
        composite = risk_report.get("composite_risk","low")
        decision_type = DECISION_POLICY.get(composite,"SAFE")

        baseline = {
            "decision_type": decision_type,
            "recommendation": DEFAULT_RECOMMENDATIONS.get(decision_type, ""),
            "explanation": risk_report.get("explanation", ""),
            "confidence": self._compute_confidence(risk_report),
        }

        try:
            llm_result = self._llm_tool.reason_decision(baseline,
                                                        risk_report,
                                                        state.get("scene_summary",{}),
                                                        state.get("detected_objects",[]))
            
            baseline['recommendation'] = llm_result.get("recommendation",baseline['recommendation'])
            baseline["explanation"] = llm_result.get("explanation",baseline["explanation"])

        except Exception as e:
            self.logger.warning(f"LLM decision reasoning failed: {e}")

        self.logger.info(f"Decision: {baseline['decision_type']}")
        
        return {"decision":baseline}

    def _compute_confidence(self, risk_report: dict) -> float:
        """Simple confidence heuristic based on risk clarity.

        Higher confidence when risks are either very low or very high
        (clear-cut situations). Lower confidence for ambiguous mid-range.
        """
        score_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        scores = [
            score_map.get(risk_report.get(k, "low"), 1)
            for k in ["collision_risk", "pedestrian_risk", "lane_risk"]
        ]
        max_score = max(scores)

        if max_score >= 4:
            return 0.95
        if max_score >= 3:
            return 0.85
        if max_score >= 2:
            return 0.75
        return 0.90