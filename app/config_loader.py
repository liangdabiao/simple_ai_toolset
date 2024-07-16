import json
import os
import logging
logger = logging.getLogger("AppLogger")

def load_config():
    config_path = os.path.join("app", "config.json")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        return None

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from config file at {config_path}")
    except Exception as e:
        # Catching other unexpected exceptions and logging them.
        logger.error(f"An unexpected error occurred while loading config: {e}")

    return None  # return None if any exception occurred.

config = load_config()
