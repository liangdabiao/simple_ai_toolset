# optimized
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import json


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage()
        }

        # Include stack trace if it exists
        if record.exc_info:
            log_record["stack_trace"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)


def setup_handler(log_file, level):
    formatter = JsonFormatter()
    handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=30
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler


def create_log_directories():
    base_path = "log"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    log_paths = [f"{base_path}/info", f"{base_path}/warning", f"{base_path}/error"]
    for path in log_paths:
        os.makedirs(path, exist_ok=True)


def initialize_logger():
    create_log_directories()
    
    logger = logging.getLogger("AppLogger")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    if info_handler := setup_handler("log/info/info.log", logging.INFO):
        logger.addHandler(info_handler)

    if warning_handler := setup_handler("log/warning/warning.log", logging.WARNING):
        logger.addHandler(warning_handler)

    if error_handler := setup_handler("log/error/error.log", logging.ERROR):
        logger.addHandler(error_handler)

    return logger
