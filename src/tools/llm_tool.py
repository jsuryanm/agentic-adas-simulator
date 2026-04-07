import json 
from pprint import pprint

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.core.config import settings 
from src.exceptions.custom_exceptions import LLMToolException
from src.logger.custom_logger import logger 


class LLMTool:
    """Generates a short human-readable narration of the scene."""
    def __init__(self):
        self._llm = None 
        self._scene_chain = None
        self._risk_chain = None 
        self._decision_chain = None 
    
    def _get_llm(self) -> ChatOpenAI:
        """Lazy loading to initializing OpenAI ChatModel"""
        if self._llm is not None:
            return self._llm
        
        if not settings.OPENAI_API_KEY:
            raise LLMToolException("OPENAI_API_KEY is not set - cannot instantiate LLM")

        self._llm = ChatOpenAI(model=settings.OPENAI_MODEL,
                               temperature=settings.LLM_TEMPERATURE,
                               max_completion_tokens=settings.LLM_MAX_TOKENS,
                               api_key=settings.OPENAI_API_KEY)
        
        logger.info(f"LLM model instantiated: {settings.OPENAI_MODEL}")
        
        return self._llm 
    
    def _get_scene_chain(self):
        """Lazy load scene chain"""

        if self._scene_chain is not None:
            return self._scene_chain
        
        llm = self._get_llm()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the reasoning module inside an ADAS (Advanced Driver Assistance Systems) Scene Understanding Agent. "
                "You receive raw object detections and a rule-based scene summary. "
                "Your job is to identify contextual patterns that simple rules miss.\n\n"
                "Reason about:\n"
                "- Spatial relationships between objects (e.g. pedestrian between parked cars)\n"
                "- Behavioural inferences (e.g. cyclist weaving, vehicle braking)\n"
                "- Environmental hazards implied by the combination of objects\n"
                "- Whether the rule-based summary under- or over-estimates any risk factor\n\n"
                "Respond ONLY with a JSON object — no markdown, no backticks:\n"
                '{{\n'
                '  "context_notes": ["<insight 1>", "<insight 2>"],\n'
                '  "narration": "<2 sentence driver-perspective summary>"\n'
                '}}',
            ),
            (
                "human",
                "Detected objects:\n{detected_objects}\n\n"
                "Rule-based scene summary:\n{scene_summary}\n\n"
                "Lane analysis:\n{lane_analysis}\n\n"
                "Provide your contextual reasoning:",
            )
        ])
    
        self._scene_chain = prompt | llm | JsonOutputParser()
        logger.info("Scene reasoning chain built")
        return self._scene_chain
    
    def reason_scene(self,
                     scene_summary: dict,
                     detected_objects: list,
                     lane_analysis: dict | None) -> dict:
        """LLM reasons about the scene and returns enrichment data.

        Returns:
            dict with "context_notes" (list[str]) and "narration" (str).
            On failure, returns rule-based fallback.
        """

        if not settings.LLM_ENABLED:
            logger.info("LLM disabled using rule-based scene output")
            return self._scene_fallback(scene_summary)
        
        try:
            chain = self._get_scene_chain()
            parsed = chain.invoke({
                "detected_objects": json.dumps(detected_objects, indent=2, default=str),
                "scene_summary": json.dumps(scene_summary, indent=2, default=str),
                "lane_analysis": json.dumps(lane_analysis, indent=2, default=str),
            })

            result = {"context_notes":parsed.get("context_notes",[]),
                      "narration": parsed.get("narration",[])}
            
            logger.info(f"Scene reasoning: {len(result["context_notes"])} insights generated")
            return result 
        
        except LLMToolException:
            raise 

        except Exception as e:
            logger.warning(f"Scene reasoning failed  using fallback: {e}")
            return self._scene_fallback(scene_summary)
        
    def _scene_fallback(self,summary: dict) -> dict:
        """Template fallback when the  LLM is unavailable"""
        parts = []
        density = summary.get("traffic_density","unknown")
        parts.append(f"Traffic is {density}")

        if summary.get("lead_vehicle_present"):
            dist = summary.get("lead_vehicle_distance", "unknown")
            parts.append(f"A lead vehicle is detected at {dist} range.")

        if summary.get("pedestrian_near_path"):
            parts.append("Pedestrian detected near the vehicle's path.")
        elif summary.get("pedestrian_present"):
            parts.append("Pedestrian visible but not in the immediate path.")

        status = summary.get("lane_status", "centered")
        if status != "centered":
            parts.append(f"Vehicle is {status.replace('_', ' ')}.")

        return {"context_notes": summary.get("context_notes", []),
                "narration": " ".join(parts) if parts else "Scene analysis complete.",}
    
    def _get_risk_chain(self):
        if self._risk_chain is not None:
            return self._risk_chain
        
        llm = self._get_llm()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the reasoning module inside an ADAS (Advanced Driver Assistance Systems) Risk Assessment Agent. "
                "You receive a rule-based risk report, the scene summary, and raw detections. "
                "The rules score collision, pedestrian, and lane risk independently — "
                "but they CANNOT cross-reference factors.\n\n"
                "Your job:\n"
                "1. Evaluate whether any risk is UNDER-scored by checking cross-factor "
                "   interactions (e.g. drifting toward the same side a cyclist is on).\n"
                "2. You may adjust a risk UP by one level if justified. "
                "   You must NEVER lower a risk — rules are the safety floor.\n"
                "3. Recompute composite risk if you changed any individual score using these rules:\n"
                "   - Any CRITICAL → composite CRITICAL\n"
                "   - 2+ HIGH → composite CRITICAL\n"
                "   - 1 HIGH → composite HIGH\n"
                "   - 2+ MEDIUM → composite HIGH\n"
                "   - else → max individual risk\n\n"
                "Respond ONLY with a JSON object — no markdown, no backticks:\n"
                '{{\n'
                '  "collision_risk": "<low|medium|high|critical>",\n'
                '  "pedestrian_risk": "<low|medium|high|critical>",\n'
                '  "lane_risk": "<low|medium|high|critical>",\n'
                '  "composite_risk": "<low|medium|high|critical>",\n'
                '  "adjustments_made": ["<what you changed and why>"],\n'
                '  "explanation": "<2 sentence reasoning covering all factors>"\n'
                '}}',
            ),
            (
                "human",
                "Rule-based risk report:\n{risk_report}\n\n"
                "Scene summary:\n{scene_summary}\n\n"
                "Detected objects:\n{detected_objects}\n\n"
                "Validate and adjust if needed:",
            ),
        ])

        self._risk_chain = prompt | llm | JsonOutputParser()
        logger.info("Risk validation chain built")
        return self._risk_chain
    
    def validate_risk(self,
                       risk_report: dict,
                       scene_summary: dict,
                       detected_object: list) -> dict:
        """LLM validates rule-based risk and may escalate (never downgrade).

        Returns:
            dict with validated/adjusted risk fields.
            On failure, returns the original rule-based report unchanged.
        """

        if not settings.LLM_ENABLED:
            logger.info("LLM disabled - using rule based risk only.")
            return risk_report
        
        try:
            chain = self._get_risk_chain()
            
            parsed = chain.invoke({"risk_report":json.dumps(risk_report,indent=2,default=str),
                                   "scene_summary":json.dumps(scene_summary,indent=2,default=str),
                                   "detected_objects":json.dumps(detected_object,indent=2,default=str)})
            
            # safety guard llm must never lower a risk score 
            validated = self._enforce_floor(risk_report,parsed)

            adjustments = validated.get("adjustment_made",[])
            if adjustments:
                logger.info(f"Risk adjusted by LLM: {adjustments}")
            else:
                logger.info("Risk validated by LLM no adjustments")
            
            return validated
        
        except LLMToolException:
            raise 

        except Exception as e:
            logger.warning(f"Risk validation failed, keeping rule-based: {e}")
            return risk_report

    def _enforce_floor(self, baseline: dict, llm_output: dict) -> dict:
        """Guarantee the LLM never downgrades a risk level.

        If the LLM returns a lower score for any dimension, we keep
        the original rule-based score instead.
        """
        score_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        risk_keys = ["collision_risk", "pedestrian_risk", "lane_risk"]

        result = {**baseline}  # start with rule-based as default

        for key in risk_keys:
            baseline_score = score_map.get(baseline.get(key, "low"), 1)
            llm_score = score_map.get(llm_output.get(key, "low"), 1)
            # Keep whichever is higher (LLM can escalate, never lower)
            result[key] = llm_output[key] if llm_score >= baseline_score else baseline[key]

        # Recompute composite from the enforced individual scores
        result["composite_risk"] = self._recompute_composite(result["collision_risk"],
                                                             result["pedestrian_risk"],
                                                             result["lane_risk"])

        # Keep LLM explanation and adjustments if present
        result["explanation"] = llm_output.get("explanation", baseline.get("explanation", ""))
        result["adjustments_made"] = llm_output.get("adjustments_made", [])
        result["primary_driver"] = baseline.get("primary_driver", "")

        return result
    
    def _recompute_composite(self, collision: str, pedestrian: str, lane: str) -> str:
        """Mirror the RiskTool composite rules to recompute after LLM adjustment."""
        levels = [collision, pedestrian, lane]

        if "critical" in levels:
            return "critical"
        high_count = sum(1 for l in levels if l == "high")
        if high_count >= 2:
            return "critical"
        if high_count == 1:
            return "high"
        med_count = sum(1 for l in levels if l == "medium")
        if med_count >= 2:
            return "high"

        score_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        scores = [score_map.get(l, 1) for l in levels]
        reverse_map = {1: "low", 2: "medium", 3: "high", 4: "critical"}
        return reverse_map.get(max(scores), "low")
    
    def _get_decision_chain(self):
        if self._decision_chain is not None:
            return self._decision_chain

        llm = self._get_llm()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the reasoning module inside an ADAS (Advanced Driver Assistance Systems) Decision Agent. "
                "You receive the full pipeline context: detections, scene summary, "
                "risk report, and a baseline decision from policy rules.\n\n"
                "Your job:\n"
                "- Write a situation-specific recommendation that references the "
                "  actual objects and hazards in THIS scene (not generic advice).\n"
                "- Write a clear explanation of WHY this decision was made, "
                "  tracing back through the reasoning chain.\n"
                "- You must NOT change the decision_type — the policy mapping is fixed.\n\n"
                "Respond ONLY with a JSON object — no markdown, no backticks:\n"
                '{{\n'
                '  "recommendation": "<specific actionable advice for this scene>",\n'
                '  "explanation": "<2-4 sentence reasoning chain from detection to decision>"\n'
                '}}',
            ),
            (
                "human",
                "Baseline decision:\n{baseline_decision}\n\n"
                "Risk report:\n{risk_report}\n\n"
                "Scene summary:\n{scene_summary}\n\n"
                "Detected objects:\n{detected_objects}\n\n"
                "Provide situation-specific recommendation:",
            ),
        ])

        self._decision_chain = prompt | llm | JsonOutputParser()
        logger.info("Decision reasoning chain built (with JsonOutputParser)")
        return self._decision_chain
    
    def reason_decision(self,
                        baseline_decision: dict,
                        risk_report: dict,
                        scene_summary: dict,
                        detected_objects: list) -> dict:
        
        """LLM writes situation-specific recommendation and explanation.

        The decision_type is NEVER changed — only recommendation and
        explanation text are enriched.

        Returns:
            dict with "recommendation" and "explanation" strings.
            On failure, returns the baseline decision's text unchanged.
        """
        """LLM writes situation-specific recommendation and explanation.

        The decision_type is NEVER changed — only recommendation and
        explanation text are enriched.

        Returns:
            dict with "recommendation" and "explanation" strings.
            On failure, returns the baseline decision's text unchanged.
        """
        if not settings.LLM_ENABLED:
            logger.info("LLM disabled — using policy-based decision only")
            return {
                "recommendation": baseline_decision.get("recommendation", ""),
                "explanation": baseline_decision.get("explanation", ""),
            }

        try:
            chain = self._get_decision_chain()
            parsed = chain.invoke({
                "baseline_decision": json.dumps(baseline_decision, indent=2, default=str),
                "risk_report": json.dumps(risk_report, indent=2, default=str),
                "scene_summary": json.dumps(scene_summary, indent=2, default=str),
                "detected_objects": json.dumps(detected_objects, indent=2, default=str),
            })

            result = {
                "recommendation": parsed.get(
                    "recommendation",
                    baseline_decision.get("recommendation", ""),
                ),
                "explanation": parsed.get(
                    "explanation",
                    baseline_decision.get("explanation", ""),
                ),
            }
            logger.info("Decision enriched by LLM with situation-specific reasoning")
            return result

        except LLMToolException:
            raise
        except Exception as e:
            logger.warning(f"Decision reasoning failed, keeping baseline: {e}")
            return {
                "recommendation": baseline_decision.get("recommendation", ""),
                "explanation": baseline_decision.get("explanation", ""),
            }


# if __name__ == "__main__":

#     llm_tool = LLMTool()

#     scene_summary = {
#         "lead_vehicle_present":True,
#         "lead_vehicle_distance":"mid",
#         "pedestrian_present":True,
#         "pedestrian_near_path":False,
#         "traffic_density":"light",
#         "lane_status":"centered",
#         "context_notes":[]
#     }

#     detected_objects = [
#         {
#             "label":"car",
#             "position":"center",
#             "distance":"mid"
#         },
#         {
#             "label":"pedestrian",
#             "position":"left",
#             "distance":"near"
#         }
#     ]

#     lane_analysis = {
#         "lateral_offset":0.05,
#         "road_detected":True
#     }

#     print("\n===== SCENE REASONING =====")

#     scene_reasoning = llm_tool.reason_scene(
#         scene_summary,
#         detected_objects,
#         lane_analysis
#     )

#     pprint(scene_reasoning)

#     risk_report = {
#         "collision_risk":"medium",
#         "pedestrian_risk":"medium",
#         "lane_risk":"low",
#         "composite_risk":"medium",
#         "primary_driver":"collision_risk",
#         "explanation":"baseline"
#     }

#     print("\n===== RISK VALIDATION =====")

#     validated = llm_tool.validate_risk(
#         risk_report,
#         scene_summary,
#         detected_objects
#     )

#     pprint(validated)

#     baseline_decision = {
#         "decision_type":"WARNING",
#         "recommendation":"Reduce speed",
#         "explanation":"High risk detected"
#     }

#     print("\n===== DECISION REASONING =====")

#     decision = llm_tool.reason_decision(
#         baseline_decision,
#         risk_report,
#         scene_summary,
#         detected_objects
#     )

#     pprint(decision)