from langgraph.graph import StateGraph,START,END

from src.pipelines.state import ADASState,initial_state
from src.agents.perception_agent import PerceptionAgent
from src.agents.scene_agent import SceneAgent
from src.agents.risk_agent import RiskAgent
from src.agents.decision_agent import DecisionAgent

from loguru import logger 


_perception = PerceptionAgent()
_scene = SceneAgent()
_risk = RiskAgent()
_decision = DecisionAgent()


def perception_node(state: ADASState) -> dict:
    return _perception.run(state)

def scene_node(state: ADASState) -> dict:
    return _scene.run(state)

def decision_node(state: ADASState) -> dict:
    return _decision.run(state)

def risk_node(state: ADASState) -> dict:
    return _risk.run(state)


def build_graph():
    """Construct the ADAS LangGraph pipeline.

    Flow: perception → scene → risk → decision → END
    """

    graph = StateGraph(ADASState)

    graph.add_node("perception",perception_node)
    graph.add_node("scene",scene_node)
    graph.add_node("decision",decision_node)
    graph.add_node("risk",risk_node)

    graph.add_edge(START,"perception")
    graph.add_edge("perception","scene")
    graph.add_edge("scene","risk")
    graph.add_edge("risk","decision")
    graph.add_edge("decision",END)
    return graph.compile()

def run_pipeline(image_path: str,frame_number: int = 0) -> ADASState:
    """Run the full ADAS pipeline on a single image.

    Args:
        image_path: Path to the driving scene image.
        frame_number: Frame index (for video processing).

    Returns:
        Final ADASState dict with all agent outputs.
    """
    state = initial_state(image_path, frame_number)
    pipeline = build_graph()
    result = pipeline.invoke(state)
    logger.info(f"Pipeline complete for frame {frame_number}")
    return result