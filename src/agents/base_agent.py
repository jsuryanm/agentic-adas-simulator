import time 
from abc import ABC,abstractmethod
from loguru import logger 

from src.pipelines.state import ADASState

class BaseAgent(ABC):
    """All agents inherit from this.

    Subclasses implement `_process(state)` which returns a dict of
    state updates.  The base class wraps it with timing and error
    handling, then returns the update dict to LangGraph.
    """

    
