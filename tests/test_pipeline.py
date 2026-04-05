import pytest
from src.models.schemas import (
    DetectedObject, LaneAnalysis, SceneSummary,
    RiskReport, RiskLevel, LaneStatus, Decision, DecisionType,
)
from src.tools.scene_tool import SceneTool
from src.tools.risk_tool import RiskTool
from src.agents.decision_agent import DECISION_POLICY



class TestSchemas:

    def test_detected_object(self):
        obj = DetectedObject(
            label="car", confidence=0.9,
            position="center", distance="near",
            bbox=[100, 200, 300, 400],
        )
        assert obj.label == "car"
        d = obj.model_dump()
        assert "bbox" in d

    def test_risk_level_scores(self):
        assert RiskLevel.LOW.score() == 1
        assert RiskLevel.CRITICAL.score() == 4

    def test_lane_status_values(self):
        assert LaneStatus.CENTERED.value == "centered"
        assert LaneStatus.DRIFTING_LEFT.value == "drifting_left"



class TestSceneTool:

    def setup_method(self):
        self.tool = SceneTool()

    def test_empty_detections(self):
        summary = self.tool.analyse([], None)
        assert summary.lead_vehicle_present is False
        assert summary.pedestrian_present is False
        assert summary.traffic_density == "clear"

    def test_lead_vehicle_detection(self):
        objects = [
            {"label": "car", "confidence": 0.9,
             "position": "center", "distance": "near",
             "bbox": [100, 200, 300, 400]},
        ]
        summary = self.tool.analyse(objects, None)
        assert summary.lead_vehicle_present is True
        assert summary.lead_vehicle_distance == "near"

    def test_pedestrian_near_path(self):
        objects = [
            {"label": "pedestrian", "confidence": 0.8,
             "position": "center", "distance": "mid",
             "bbox": [500, 220, 580, 420]},
        ]
        summary = self.tool.analyse(objects, None)
        assert summary.pedestrian_present is True
        assert summary.pedestrian_near_path is True

    def test_traffic_density_congested(self):
        objects = [
            {"label": "car", "confidence": 0.9,
             "position": p, "distance": "mid",
             "bbox": [0, 0, 100, 100]}
            for p in ["left", "center", "right",
                      "left", "center", "right",
                      "left", "center"]
        ]
        summary = self.tool.analyse(objects, None)
        assert summary.traffic_density == "congested"

    def test_lane_drift(self):
        lane = {"lateral_offset": 0.5, "road_detected": True,
                "road_coverage": 1.0, "lane_width_px": 800}
        summary = self.tool.analyse([], lane)
        assert summary.lane_status == LaneStatus.DRIFTING_RIGHT



class TestRiskTool:

    def setup_method(self):
        self.tool = RiskTool()

    def test_all_clear(self):
        summary = {
            "lead_vehicle_present": False,
            "pedestrian_present": False,
            "lane_status": "centered",
            "traffic_density": "clear",
        }
        report = self.tool.assess(summary)
        assert report.composite_risk == RiskLevel.LOW

    def test_critical_pedestrian(self):
        summary = {
            "lead_vehicle_present": False,
            "pedestrian_present": True,
            "pedestrian_near_path": True,
            "lane_status": "centered",
            "traffic_density": "clear",
        }
        report = self.tool.assess(summary)
        assert report.pedestrian_risk == RiskLevel.CRITICAL
        assert report.composite_risk == RiskLevel.CRITICAL

    def test_composite_escalation(self):
        """Two MEDIUM risks should escalate to HIGH composite."""
        summary = {
            "lead_vehicle_present": True,
            "lead_vehicle_distance": "near",
            "pedestrian_present": True,
            "pedestrian_near_path": False,
            "lane_status": "centered",
            "traffic_density": "light",
        }
        report = self.tool.assess(summary)
        # collision=MEDIUM, pedestrian=MEDIUM → composite=HIGH
        assert report.composite_risk == RiskLevel.HIGH



class TestDecisionPolicy:

    def test_policy_mapping(self):
        assert DECISION_POLICY["low"] == "SAFE"
        assert DECISION_POLICY["medium"] == "ADVISORY"
        assert DECISION_POLICY["high"] == "WARNING"
        assert DECISION_POLICY["critical"] == "CRITICAL"
