import yaml
import logging
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        raise