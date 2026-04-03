

class ADASBaseException(Exception):
    """Root exception for the entire ADAS simulator.
    Catch this if you want to handle any ADAS error in one place."""
    pass


# Tool Exceptions 
class DetectionToolException(ADASBaseException):
    """Raised when YOLO inference fails (bad image path, model not loaded, etc.)"""
    pass


class SceneToolException(ADASBaseException):
    """Raised when scene analysis logic encounters unexpected input."""
    pass


class RiskToolException(ADASBaseException):
    """Raised when risk calculation receives malformed scene data."""
    pass


class LLMToolException(ADASBaseException):
    """Raised when the LLM tool fails (API error, timeout, bad response, etc.)"""
    pass

class DepthToolException(ADASBaseException):
    """Raised when depth estimation fails (model load, inference, etc.)"""
    pass
 
class LaneToolException(ADASBaseException):
    """Raised when lane detection fails (ONNX load, inference, decode, etc.)"""
    pass


# Agent Exceptions

class PerceptionAgentException(ADASBaseException):
    """Raised when the Perception Agent cannot complete its task."""
    pass


class SceneAgentException(ADASBaseException):
    """Raised when the Scene Understanding Agent fails."""
    pass


class RiskAgentException(ADASBaseException):
    """Raised when the Risk Assessment Agent fails."""
    pass


class DecisionAgentException(ADASBaseException):
    """Raised when the Decision Agent cannot produce a recommendation."""
    pass


# Pipeline Exceptions 

class PipelineException(ADASBaseException):
    """Raised when the LangGraph pipeline itself fails to run."""
    pass


class ImageLoadException(ADASBaseException):
    """Raised when an image or video frame cannot be read from disk."""
    pass


class MCPServerException(ADASBaseException):
    """Raised when the MCP tool server encounters an error."""
    pass