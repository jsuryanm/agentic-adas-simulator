import sys 
from loguru import logger 
from src.core.config import settings

def setup_logger():
    logger.remove()

    logger.add(sys.stdout,
               level=settings.LOG_LEVEL,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {function} | {message}")

    logger.add(settings.LOG_FILE,
            rotation="10 MB",
            retention="10 days",
            compression="zip",
            level=settings.LOG_LEVEL)

setup_logger()
    
