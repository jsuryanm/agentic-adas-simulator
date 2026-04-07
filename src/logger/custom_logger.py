from loguru import logger 
from src.core.config import settings


logger.add(settings.LOG_FILE,
           level=settings.LOG_FILE,
           rotation="10 MB",
           retention="30 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {function} | {message}")