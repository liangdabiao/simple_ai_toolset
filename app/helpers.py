import logging
from enum import Enum, auto
import inspect 
import traceback
import sys

class LogLevel(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()

logger = logging.getLogger("AppLogger")


def record_info_log(message):
    logger.info(message)

def record_log(exception : Exception,module_name,function_name, level: LogLevel):
    if not isinstance(level, LogLevel):
        raise ValueError("Invalid log level")

    log_functions = {
        LogLevel.ERROR: logger.error,
        LogLevel.WARNING: logger.warning,
        LogLevel.INFO: logger.info,
    }

    try:
        exc_type = sys.exc_info()
        log_func = log_functions.get(level, logger.info)
        if level in [LogLevel.ERROR, LogLevel.WARNING]:
            log_func(f"Error in {module_name}:{function_name}, Message: {str(exception)}, Type: {exc_type}", exc_info=True)

    except Exception as e:
        logger.error(f"Failed to record log: {e}", exc_info=True)

def get_calling_function_name():
    # Use inspect.stack() to get the previous frame and extract its function name
    function_name = inspect.stack()[1].function
    return function_name


def get_calling_module_name():
    # Get the frame of the caller
    caller_frame = inspect.stack()[1]
    # Get the module of the caller
    module = inspect.getmodule(caller_frame[0])
    # Return the module's name
    return module.__name__ if module is not None else None

