from typing import Optional,Tuple,List 
import os
import numpy as np 

import cv2 
import torch 

from PIL import Image
from transformers import AutoImageProcessor,AutoModelForDepthEstimation

from src.tools.detection_tool import DetectionTool
from src.core.config import settings 
from src.exceptions.custom_exceptions import DepthToolException,ImageLoadException
from src.logger.custom_logger import logger 

class DepthTool:
    """
    Wraps a monocular depth estimation model and provides two services:
 
    1. estimate()         → full depth map for a frame
    2. enrich_detections() → upgrade DetectedObject distance labels in-place
    """
 
    def __init__(self):
        self._processor = None 
        self._model = None
        
    
    def load_model(self) -> None:
        """Lazy load model on first use"""
        if self._model is not None:
            return 
        
        try:
            self._processor = AutoImageProcessor.from_pretrained(settings.DEPTH_MODEL)
            self._model = AutoModelForDepthEstimation.from_pretrained(settings.DEPTH_MODEL)
            self._model.eval()

            device = settings.DEPTH_DEVICE
            self._model.to(device)
            logger.info(f"Depth model loaded on {device}")
        
        except Exception as e:
            raise DepthToolException(f"Failed to load depth model: {e}")
        
    
    def estimate(self,image_path: str) -> np.ndarray:
        """
        Run depth estimation on a single frame.
 
        Args:
            image_path: Path to the image file.
 
        Returns:
            depth_map: np.ndarray of shape (H, W), float32, 
            normalised to [0, 1].
            Higher values = closer to the camera.
 
        Raises:
            ImageLoadException: If the image cannot be read.
            DepthToolException: If inference fails.
        """
        frame = cv2.imread(image_path)

        if frame is None:
            raise ImageLoadException(f"Cannot read image: {image_path}")
        
        self.load_model()

        try:
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            inputs = self._processor(images=pil_image,return_tensors="pt")
            inputs = {key:val.to(settings.DEPTH_DEVICE) for key,val in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
            
            depth = outputs.predicted_depth.squeeze(0).cpu().numpy()

            d_min,d_max = depth.min(),depth.max()
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)

            img_h,img_w = frame.shape[:2]

            depth_map = cv2.resize(depth_norm,
                                   (img_w,img_h),
                                   interpolation=cv2.INTER_LINEAR)
            logger.info(f"Depth map computed: shape: {depth_map.shape}, range: [{depth_map.min():.3f},{depth_map.max():.3f}]")
            
            return depth_map.astype(np.float32)

        except Exception as e:
            raise DepthToolException(f"Depth Estimation failed: {e}")
        
    def enrich_detections(self,
                          detected_objects: List[dict],
                          depth_map: np.ndarray,
                          img_dims: Tuple[int,int]) -> List[dict]:
        """
        Upgrade the 'distance' field of each detection using the depth map.
 
        This replaces the bbox-height heuristic with learned depth.
        The original detection dicts are returned with updated 'distance'
        and an added 'depth_value' field.
 
        Args:
            detected_objects: List of DetectedObject.to_dict() results.
            depth_map:        Normalised depth map (H, W), values in [0, 1].
            img_dims:         (width, height) of the original image.
 
        Returns:
            The same list with updated distance labels.
        """
        img_w,img_h = img_dims 
        map_h,map_w = depth_map.shape[:2]

        for obj in detected_objects:
            depth_val = self._sample_depth(obj["bbox"],depth_map,img_w,img_h,map_w,map_h)
            obj["depth_value"] = round(float(depth_val),4)
            obj["distance"] = self._depth_to_label(depth_val)
        
        logger.info(f"Enriched {len(detected_objects)} detections with depth values")
        return detected_objects
    

    def _sample_depth(self,
                      bbox: List[float],
                      depth_map: np.ndarray,
                      img_w: int,
                      img_h: int,
                      map_w: int,
                      map_h: int,) -> float:
        """
        Sample the median depth in a small patch around the object's
        bounding-box centre.  Median is robust to depth-edge artifacts.
        """
        x1, y1, x2, y2 = bbox
 
        # Centre of bounding box, scaled to depth-map coordinates
        cx = int((x1 + x2) / 2 * map_w / img_w)
        cy = int((y1 + y2) / 2 * map_h / img_h)
 
        # Sample radius: 10 % of the smaller bbox dimension (at least 3 px)
        bbox_size = min(x2 - x1, y2 - y1)
        r = max(3, int(bbox_size * 0.1 * map_w / img_w))
 
        # Clamp to map bounds
        y_lo = max(0, cy - r)
        y_hi = min(map_h, cy + r)
        x_lo = max(0, cx - r)
        x_hi = min(map_w, cx + r)
 
        patch = depth_map[y_lo:y_hi, x_lo:x_hi]
        if patch.size == 0:
            return 0.5  # safe fallback
 
        return float(np.median(patch))
 
    def _depth_to_label(self, depth_value: float) -> str:
        """
        Convert normalised depth to a distance category.
 
        Depth Anything V2 outputs *inverse depth* (disparity-like):
            higher value → object is closer.
        """
        if depth_value > settings.DEPTH_NEAR_THRESHOLD:
            return "near"
        elif depth_value > settings.DEPTH_MID_THRESHOLD:
            return "mid"
        return "far"
    
# if __name__ == "__main__":

#     BASE_DIR = os.path.dirname(__file__)
#     img_path = os.path.join(BASE_DIR,"images","test.jpg")

#     depth_tool = DepthTool()
#     detection_tool = DetectionTool()

#     # Run detection
#     detected, dims = detection_tool.detect(img_path)

#     # Compute depth
#     depth_map = depth_tool.estimate(img_path)

#     # Convert DetectedObject -> dict (LangGraph rule compliance)
#     detected_dicts = [obj.model_dump() for obj in detected]

#     # Enrich with depth
#     enriched = depth_tool.enrich_detections(detected_dicts,
#                                             depth_map,
#                                             dims)

#     # Load image for drawing
#     frame = cv2.imread(img_path)

#     # Create depth visualization
#     depth_vis = (depth_map * 255).astype("uint8")
#     depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

#     # Ensure same size
#     depth_vis = cv2.resize(depth_vis,
#                            (frame.shape[1], frame.shape[0]))

#     # Draw detections
#     for obj in enriched:

#         x1,y1,x2,y2 = map(int,obj["bbox"])

#         if obj["distance"] == "near":
#             color = (0,0,255)

#         elif obj["distance"] == "mid":
#             color = (0,165,255)

#         else:
#             color = (0,255,0)

#         label = f'{obj["label"]} {obj["distance"]} ({obj["depth_value"]:.2f})'

#         cv2.rectangle(frame,
#                       (x1,y1),
#                       (x2,y2),
#                       color,
#                       2)

#         cv2.putText(frame,
#                     label,
#                     (x1,y1-8),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     color,
#                     2)

#         # Draw depth sampling point (debug)
#         cx = int((x1+x2)/2)
#         cy = int((y1+y2)/2)

#         cv2.circle(frame,
#                    (cx,cy),
#                    4,
#                    (255,255,0),
#                    -1)

#     # Create overlay
#     overlay = cv2.addWeighted(frame,
#                               0.7,
#                               depth_vis,
#                               0.3,
#                               0)

#     # Create resizable windows
#     cv2.namedWindow("ADAS Detections",cv2.WINDOW_NORMAL)
#     cv2.namedWindow("Depth Map",cv2.WINDOW_NORMAL)
#     cv2.namedWindow("Depth Overlay",cv2.WINDOW_NORMAL)

#     # Show windows
#     cv2.imshow("ADAS Detections",frame)
#     cv2.imshow("Depth Map",depth_vis)
#     cv2.imshow("Depth Overlay",overlay)

#     cv2.waitKey(0)

#     cv2.destroyAllWindows()