from typing import List 
from src.core.config import settings 
from loguru import logger 

from src.models.schemas import DetectedObject
from src.exceptions.custom_exceptions import DetectionToolException,ImageLoadException

import cv2 
import os
from ultralytics import YOLO 

class DetectionTool:

    RELEVANT_CLASSES = {0:"person",
                        1:"bicycle",
                        2:"car",
                        3:"motorcycle",
                        5:"bus",
                        7:"truck",
                        9:"traffic_light",
                        11:"stop_sign"}
    
    def __init__(self):
        self._model = None 
    
    def load_model(self):
        if self._model is None:
            logger.info(f"Loading YOLO model: {settings.YOLO_MODEL}")

            try:
                self._model = YOLO(settings.YOLO_MODEL)
                logger.info(f"{settings.YOLO_MODEL} model loaded successfully")
            
            except Exception as e:
                raise DetectionToolException(f"Failed to load YOLO model: {e}") 
            
    def detect(self,image_path: str) -> tuple[List[DetectedObject],tuple[int,int]]:
        frame = cv2.imread(image_path)

        if frame is None:
            raise ImageLoadException(f"Failed to load image: {image_path}")
        
        img_height,img_width = frame.shape[:2]        
        logger.info(f"Running object detection on {image_path} with shape:({img_height},{img_width})")

        self.load_model()

        try:                
            results = self._model(frame,
                                  conf=settings.CONFIDENCE_THRESHOLD,
                                  verbose=False,
                                  device=settings.YOLO_DEVICE)
        except Exception as e:
            raise DetectionToolException(f"YOLO inference failed: {e}") from e          
        
        detected: List[DetectedObject] = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item()) 
                # note all outputs of boxes are pytorch tensors use item()

                if class_id not in self.RELEVANT_CLASSES:
                    continue 

                x1,y1,x2,y2 = box.xyxy.squeeze().tolist()
                confidence = float(box.conf.item())
                label = self.RELEVANT_CLASSES[class_id]

                obj = DetectedObject(label=label,
                                     confidence=round(confidence,3),
                                     position=self._get_horizontal_position(x1,x2,img_width),
                                     distance=self._get_distance_estimate(y1,y2,img_height),
                                     bbox=[float(x1),float(y1),float(x2),float(y2)])
                
                detected.append(obj)

        detected = self._filter_detections(detected,img_width,img_height)
        logger.info(f"Detected {len(detected)} relevant objects.")

        return detected,(img_width,img_height)
    
    def _get_horizontal_position(self,
                                 x1: float,
                                 x2: float,
                                 img_width: int) -> str:
        x_center = (x1 + x2) / 2 
        ratio =  x_center / img_width # normalization step
        #  we will divide the img into 3 regions [0    -   0.33    -   0.67    -   1
                                        # (left-range[0-0.33])    (center)    (right range[0.67-1.0])]
        if ratio < 0.33: 
            return "left"
        
        elif ratio > 0.67:
            return "right"
        
        return "center"
    
    def _get_distance_estimate(self,
                               y1: float,
                               y2: float,
                               img_height: int) -> str:
        """Estimate object distance using bounding box size"""
        bbox_height_ratio =  (y2 - y1) / img_height
        # closer objects appear bigger

        if bbox_height_ratio > settings.NEAR_DISTANCE_THRESHOLD:
            return "near"
        
        elif bbox_height_ratio > settings.MID_DISTANCE_THRESHOLD:
            return "mid"
        
        return "far"
    
    def _filter_detections(self,
                           detections: List[DetectedObject],
                           img_width: int,
                           img_height: int) -> List[DetectedObject]:
        """
        Fixes YOLO mistakes using reasoning
        Main tasks:
        1. Supress false persons
        2. Merge rider detections
        3. Relabel pedestrians 
        4. Infer missed riders
        """
        persons = [d for d in detections if d.label == "person"]
        vehicles = [d for d in detections if d.label != "person"]

        merged = []
        consumed_persons = set()
        consumed_vehicles = set()

        for p_idx, person in enumerate(persons):
            best_match = None
            best_iou = 0.0
            match_type = None

            for v_idx, vehicle in enumerate(vehicles):
                if v_idx in consumed_vehicles:
                    continue

                iou = self._calculate_iou(person.bbox, vehicle.bbox)
                center = self._center_inside(person.bbox, vehicle.bbox)

                if not (center or iou > 0.15):
                    continue

                # Person overlaps a large vehicle -> suppress person only
                # This means we will only detect the car removing person
                if vehicle.label in ["car", "bus", "truck"]:
                    match_type = "suppress"
                    best_match = v_idx
                    break

                # Person overlaps motorcycle/bicycle -> merge (person + motorcycle)
                if vehicle.label in ["motorcycle", "bicycle"] and iou > best_iou:
                    best_iou = iou
                    best_match = v_idx
                    match_type = "merge"

            if match_type == "suppress":
                consumed_persons.add(p_idx)

            elif match_type == "merge" and best_match is not None:
                vehicle = vehicles[best_match]

                # Union bounding box
                union_bbox = [min(person.bbox[0], vehicle.bbox[0]),
                              min(person.bbox[1], vehicle.bbox[1]),
                              max(person.bbox[2], vehicle.bbox[2]),
                              max(person.bbox[3], vehicle.bbox[3])]

                label = "rider" if vehicle.label == "motorcycle" else "cyclist"
                confidence = max(person.confidence, vehicle.confidence)

                merged_obj = DetectedObject(label=label,
                                            confidence=round(confidence, 3),
                                            position=self._get_horizontal_position(union_bbox[0], union_bbox[2], img_width),
                                            distance=self._get_distance_estimate(union_bbox[1], union_bbox[3], img_height),
                                            bbox=union_bbox,)

                merged.append(merged_obj)
                consumed_persons.add(p_idx)
                consumed_vehicles.add(best_match)

        # --------------------------------------------------
        # Build final list
        # --------------------------------------------------
        filtered = []

        # Surviving vehicles (not merged into rider/cyclist)
        for v_idx, vehicle in enumerate(vehicles):
            if v_idx in consumed_vehicles:
                continue
            filtered.append(vehicle)

        # Surviving persons (not suppressed or merged)
        for p_idx, person in enumerate(persons):
            if p_idx in consumed_persons:
                continue

            bbox_height = person.bbox[3] - person.bbox[1]
            bbox_width = person.bbox[2] - person.bbox[0]

            if bbox_width == 0:
                continue

            aspect_ratio = bbox_height / bbox_width

            # Infer missed riders — tall narrow shapes at mid/far distance
            if aspect_ratio > 1.3 and person.distance in ["far", "mid"]:
                person.label = "rider"
                filtered.append(person)
                continue

            person.label = "pedestrian"
            filtered.append(person)

        # Add merged rider/cyclist detections
        filtered.extend(merged)

        return filtered
    
    def _calculate_iou(self,box1,box2):
        # Find the overlap rectangle 
        x1 = max(box1[0],box2[0])
        y1 = max(box1[1],box2[1])
        x2 = min(box1[2],box2[2])
        y2 = min(box1[3],box2[3])

        intersection = max(0,x2-x1) * max(0,y2-y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection

        if union == 0:
            return 0 
        
        return intersection / union
    
    def _center_inside(self,box1,box2):
        """Detect containment even when IoU is small"""
        cx = (box1[0] + box1[2]) / 2
        cy = (box1[1] + box1[3]) / 2

        return (box2[0] <= cx <= box2[2] and  box2[1] <= cy <= box2[3])
    

# if __name__ == "__main__":
#     tool = DetectionTool()

#     BASE_DIR = os.path.dirname(__file__)
#     img_path = os.path.join(BASE_DIR,"images","test.jpg")
    
#     detected, dims = tool.detect(img_path)
#     logger.info(dims)

#     for obj in detected:
#         logger.info(obj.model_dump())

#     frame = cv2.imread(img_path)
#     for obj in detected:

#         x1,y1,x2,y2 = map(int,obj.bbox)

#         if obj.label == "rider":
#             color = (0,255,255)

#         elif obj.label == "cyclist":
#             color = (255,165,0)

#         elif obj.label == "motorcycle":
#             color = (0,200,200)

#         elif obj.label == "pedestrian":
#             color = (255,0,0)

#         else:
#             color = (255,255,255)

#         cv2.rectangle(frame,
#                       (x1,y1),
#                       (x2,y2),
#                       color,2)

#         cv2.putText(frame,
#                     f"{obj.label} {obj.confidence}",
#                     (x1,y1-5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     color,
#                     2)

#     cv2.imshow("Filtered Detection",frame)
#     cv2.waitKey(0)