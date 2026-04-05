from typing_extensions import TypedDict
from operator import add 
from typing import Dict,List,Optional,Annotated

def merge_dicts(old,new):
    return {**old,**new}

class ADASState(TypedDict):
    # Inputs
    image_path: str # path to frame being analyzed 
    frame_number: int # index for video processing

    # Agent Outputs
    image_dims: dict #{"width":"int","height":"int"}
    detected_objects: Annotated[List[dict],add]
    scene_summary: Optional[dict]
    risk_report: Optional[dict]
    decision: Optional[dict]
    lane_analysis: Optional[dict]

    # metadata 
    errors: Annotated[List[str],add]
    processing_time: Annotated[dict,merge_dicts]

def initial_state(image_path: str, frame_number: int = 0) -> ADASState:
    """Create a fresh new state for new frame"""
    return ADASState(image_path=image_path,
                     frame_number=frame_number,
                     image_dims={},
                     detected_objects=[],
                     scene_summary=None,
                     risk_report=None,
                     decision=None,
                     lane_analysis=None,
                     errors=[],
                     processing_time={})
