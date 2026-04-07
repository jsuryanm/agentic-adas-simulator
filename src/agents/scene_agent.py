from src.agents.base_agent import BaseAgent
from src.tools.scene_tool import SceneTool
from src.tools.lane_tool_cv import LaneToolCV
from src.tools.llm_tool import LLMTool
from src.pipelines.state import ADASState

class SceneAgent(BaseAgent):
    """Scene Understanding Agent — converts raw detections into a semantic
    SceneSummary and runs lane analysis.

    Writes to state:
        scene_summary   (dict)
        lane_analysis   (dict)
    """
    name = "scene_agent"
    
    def __init__(self):
        super().__init__()
        self.scene_tool = SceneTool()
        self.lane_tool = LaneToolCV()
        self.llm_tool = LLMTool()

    def _process(self,state: ADASState) -> dict:
        image_path = state["image_path"]
        detected_objects = state.get("detected_objects",[])

        lane_analysis_dict = None 
        try:
            lane_result = self.lane_tool.detect_lanes(image_path)
            lane_analysis_dict = lane_result.model_dump()

        except Exception as e:
            self.logger.warning(f"Lane detection failed, continuing without: {e}")
        
        summary = self.scene_tool.analyse(detected_objects,lane_analysis_dict)
        summary_dict = summary.model_dump()

        try:
            llm_result = self.llm_tool.reason_scene(summary_dict,detected_objects,lane_analysis_dict)
            llm_notes = llm_result.get("context_notes",[])

            if llm_notes:
                summary_dict["context_notes"] = summary_dict.get("context_notes",[]) + llm_notes
            
            summary_dict["llm_narration"] = llm_result.get("narration","")
        
        except Exception as e:
            self.logger.warning(f"LLM scene reasoning failed: {e}")
            summary_dict["llm_narration"] = ""
        
        self.logger.info(f"Scene analysis complete: traffic = {summary_dict.get('traffic_density')} lane = {summary_dict.get('lane_status')}")
        
        return {"scene_summary":summary_dict,
                "lane_analysis":lane_analysis_dict}