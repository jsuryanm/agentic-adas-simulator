import os 
from typing import Optional,Tuple 

import numpy as np
import cv2 
import torch 
from torch.nn import functional as F
from transformers import (SegformerImageProcessor,
                          SegformerForSemanticSegmentation)


from PIL import Image 

from src.core.config import settings
from src.models.schemas import LaneAnalysis
from src.exceptions.custom_exceptions import (LaneToolException,
                                              ImageLoadException)
from loguru import logger   

class LaneTool:
    """Segments road surface and computes ego-vehicle lateral offset"""
    ROAD_CLASS_ID = 0 # cityscapes trainId for ROAD

    def __init__(self):
        self._processor = None 
        self._model = None 
        
    def load_model(self) -> None:
        if self._model is not None:
            return 
        
        try:
            self._processor = SegformerImageProcessor.from_pretrained(settings.LANE_MODEL)
            self._model = SegformerForSemanticSegmentation.from_pretrained(settings.LANE_MODEL)
            self._model.eval()
            self._model.to(settings.LANE_DEVICE)
            logger.info(f"Lane model {settings.LANE_MODEL} loaded on {settings.LANE_DEVICE}")
        
        except Exception as e:
            raise LaneToolException(f"Failed to loaded lane model: {e}")
        
    def detect_lanes(self,image_path: str) -> LaneAnalysis:
        """Segment the road and compute lateral offset
        Args:
            image_path: Path to the driving frame.
 
        Returns:
            LaneAnalysis with road mask info and lateral offset"""
        
        frame = cv2.imread(image_path)

        if frame is None:
            raise ImageLoadException(f"Cannot read image: {image_path}")

        img_h,img_w = frame.shape[:2]
        self.load_model()

        try:
            # runs segmentation computes the binary road mask
            road_mask = self._segment_road(frame)

            # find road boundaries from the bottom of image
            left_edge,right_edge = self._find_road_edges(road_mask,img_h,img_w)
            
            lateral_offset = self._compute_lateral_offset(left_edge,right_edge,img_w)

            lane_width = None 
            if left_edge is not None and right_edge is not None:
                lane_width = float(right_edge - left_edge)

            road_coverage = self._compute_road_coverage(road_mask,img_h)

            analysis = LaneAnalysis(lateral_offset=round(max(-1.0,min(1.0,lateral_offset)),4),
                                    lane_width_px=round(lane_width,1) if lane_width else None,
                                    road_coverage=round(road_coverage,3),
                                    road_detected=road_coverage > 0.05)
            
            logger.info(f"Lane analysis offset: {analysis.lateral_offset:.3f}, road_coverage: {analysis.road_coverage:.1%}")

            return analysis
        
        except LaneToolException:
            raise 
        except Exception as e:
            raise LaneToolException(f"Lane detection failed:{e}")
    
    def _segment_road(self,frame: np.ndarray) -> np.ndarray:
        """
        Run SegFormer and return a binary road mask (same size as input).
 
        Returns:
            road_mask: np.ndarray shape (H, W), dtype bool.
                       True = road pixel, False = not road.
        """
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)

        inputs = self._processor(images=pil_image,return_tensors="pt")
        inputs = {key:val.to(settings.LANE_DEVICE) for key,val in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs) # (shape: 1,num_classes,h/4,w/4)

        logits = F.interpolate(outputs.logits,
                                size=(frame.shape[0],frame.shape[1]),
                                mode="bilinear",
                                align_corners=False)
        
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        road_mask = (pred == self.ROAD_CLASS_ID)
        return road_mask
    
    
    def _find_road_edges(self,
                         road_mask: np.ndarray,
                         img_h: int,
                         img_w: int) -> Tuple[Optional[float],Optional[float]]:
        """
        Find the left and right road boundaries in the bottom quarter.
 
        Scans the bottom rows of the road mask to find where
        the road starts and ends horizontally.
        """
        bottom_start = int(img_h * 0.75)
        bottom_strip = road_mask[bottom_start:,:]

        column_sums = bottom_strip.sum(axis=0)
        
        threshold = bottom_strip.shape[0] * 0.3
        road_columns = np.where(column_sums > threshold)[0]

        if len(road_columns) < 2:
            return None,None 
        
        left_edge = float(road_columns[0])
        right_edge = float(road_columns[-1])
        return left_edge,right_edge


    def _compute_lateral_offset(self,
                                left_edge: Optional[float],
                                right_edge: Optional[float],
                                img_w: int,) -> float:
        """
        Compute how centered the ego vehicle is within the road.
 
        Returns:
            float in [-1, 1].
             0.0 = image centre is perfectly centred on the road
            -1.0 = image centre is at the left road edge
            +1.0 = image centre is at the right road edge
        """
        img_cx = img_w /2 

        if left_edge is not None and right_edge is not None:
            road_cx = (left_edge + right_edge) / 2 
            road_w = right_edge - left_edge

            if road_w < 1:
                return 0.0 
            
            return (img_cx - road_cx) / (road_w / 2)
        
        return 0.0 
    
    def _compute_road_coverage(self,
                              road_mask: np.ndarray,
                              img_h: int) -> float:
        """What fraction of the bottom quarter of the image is road."""
        bottom_start = int(img_h * 0.75)
        bottom_strip = road_mask[bottom_start:, :]
 
        if bottom_strip.size == 0:
            return 0.0
 
        return float(bottom_strip.sum() / bottom_strip.size)

if __name__ == "__main__":

    tool = LaneTool()

    BASE_DIR = os.path.dirname(__file__)
    img_path = os.path.join(BASE_DIR,"images","test.jpg")

    analysis = tool.detect_lanes(img_path)

    print("\n===== LANE ANALYSIS =====")
    print("Lateral offset:", analysis.lateral_offset)
    print("Lane width:", analysis.lane_width_px)
    print("Road coverage:", analysis.road_coverage)
    print("Road detected:", analysis.road_detected)


    frame = cv2.imread(img_path)

    road_mask = tool._segment_road(frame)
    h,w = frame.shape[:2]

    left_edge,right_edge = tool._find_road_edges(road_mask,
                                                 h,
                                                 w)

    # Convert mask to color overlay
    mask_vis = (road_mask.astype(np.uint8) * 255)

    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    overlay = frame.copy()

    overlay[road_mask] = (0,255,0)

    blended = cv2.addWeighted(frame,
                              0.7,
                              overlay,
                              0.3,
                              0)

    # Draw road edges
    if left_edge is not None:

        cv2.line(blended,
                 (int(left_edge),0),
                 (int(left_edge),h),
                 (255,0,0),
                 2)

    if right_edge is not None:

        cv2.line(blended,
                 (int(right_edge),0),
                 (int(right_edge),h),
                 (0,0,255),
                 2)

    # Draw image center
    cv2.line(blended,
             (w//2,0),
             (w//2,h),
             (255,255,255),
             2)

    # Draw road center
    if left_edge is not None and right_edge is not None:

        road_center = int((left_edge+right_edge)/2)

        cv2.line(blended,
                 (road_center,0),
                 (road_center,h),
                 (0,255,255),
                 2)

    cv2.putText(blended,
                f"Offset: {analysis.lateral_offset}",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2)

    cv2.imshow("Road Mask",mask_vis)
    cv2.imshow("Lane Analysis",blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()