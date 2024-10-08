import yaml
import logging
from typing import Any, Dict

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the preprocessing configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        raise