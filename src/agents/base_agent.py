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

    name: str = "base_agent"

    def __init__(self):
        self.logger = logger.bind(agent=self.name)
    
    def run(self,state: ADASState) -> dict:
        """Entry-point called by the LangGraph node.

        Returns a dict of state field updates.
        """
        self.logger.info(f"{self.name} starting (frame {state.get("frame_number","?")})")
        start = time.time()

        try:
            updates = self._process(state)
            elapsed = round(time.time() - start, 3)
            updates.setdefault("processing_time", {})[self.name] = elapsed
            self.logger.info(f"{self.name} completed in {elapsed}s")
            return updates

        except Exception as e:
            elapsed = round(time.time() - start,3)
            error_msg = f"{self.name} failed: {e}"
            self.logger.error(error_msg)
            return {"errors":[error_msg],
                    "processing_time":{self.name: elapsed}} 
    
    @abstractmethod
    def _process(self,state:ADASState) -> dict:
        """Implement the agents core logic
        Must return with dict witho only the fields this agent owns"""
        ...