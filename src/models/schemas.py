from pydantic import BaseModel,Field 
from enum import Enum 
from typing import List,Optional

class RiskLevel(str,Enum):
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def score(self) -> int:
        return {"low" : 1,
                "medium" : 2,
                "high": 3,
                "critical" : 4}[self.value]
    # self.value returns the number mapped to enum value 

class DecisionType(str,Enum):
    SAFE = "SAFE"
    ADVISORY = "ADVISORY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class DetectedObject(BaseModel):
    label: str 
    confidence: float = Field(ge=0,le=1)
    position: str 
    distance: str 
    bbox: List[float]

class LaneStatus(str,Enum):
    CENTERED = "centered"
    DRIFTING_LEFT = "drifting_left"
    DRIFTING_RIGHT = "drifting_right"

class LaneAnalysis(BaseModel):
    lateral_offset: float = Field(ge=-1,le=1) # -1 (left edge) to +1 (right edge), 0 = centred
    lane_width_px: Optional[float] = None
    road_coverage: float = Field(ge=0,le=1)
    road_detected: bool = False 

class SceneSummary(BaseModel):
    lead_vehicle_present: bool
    lead_vehicle_distance: str
    
    pedestrian_present: bool 
    pedestrian_near_path: bool 

    traffic_density: str 
    lane_status: LaneStatus
    context_notes: List[str] = Field(default_factory=list)
    llm_narration: str = ""

class RiskReport(BaseModel):
    collision_risk: RiskLevel 
    pedestrian_risk: RiskLevel 
    lane_risk: RiskLevel 
    composite_risk: RiskLevel
    primary_driver: str
    explanation: str

class Decision(BaseModel):
    decision_type: DecisionType
    recommendation: str
    explanation: str
    confidence: float = Field(ge=0,le=1) 

    

# if __name__ == "__main__":
#     risk = RiskLevel.LOW 
#     print(risk.score())