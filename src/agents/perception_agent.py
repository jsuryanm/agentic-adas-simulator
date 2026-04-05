from src.agents.base_agent import BaseAgent 
from src.tools.detection_tool import DetectionTool
from src.tools.depth_tool import DepthTool
from src.core.config import settings 
from src.pipelines.state import ADASState

class PerceptionAgent(BaseAgent):
    """Perception Agent — runs object detection (YOLO) and optional depth
        enrichment on the input image.

        Writes to state:
            detected_objects  (list of dicts)
            image_dims        (dict with width/height)
    """


    name = "perception_agent"

    def __init__(self):
        super().__init__()
        self.detection_tool = DetectionTool()
        self.depth_tool = DepthTool() if settings.DEPTH_ENABLED else None 

    def _process(self,state: ADASState) -> dict:
        image_path = state["image_path"]

        detected,(img_w,img_h) = self.detection_tool.detect(image_path)
        detected_dicts = [obj.model_dump() for obj in detected]
        
        if self.depth_tool is not None and detected_dicts:
            depth_map = self.depth_tool.estimate(image_path)
            detected_dicts = self.depth_tool.enrich_detections(detected_dicts,depth_map,(img_w,img_h))

        self.logger.info(f"Perception complete: {len(detected_dicts)} objects, shape: ({img_w},{img_h})")
        
        return {"detected_objects":detected_dicts,
                "image_dims":{"width":img_w,"height":img_h}}