import yaml
import logging
import sys
import os

def load_config(config_path):
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logging.debug("Configuration loaded successfully")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading configuration file: {e}")
        sys.exit(1)

def get_config_value(config, keys, default=None, required=False):
    value = config
    for key in keys:
        if key in value:
            value = value[key]
        else:
            if required:
                logging.error(f"Missing required configuration key: {'.'.join(keys)}")
                raise KeyError(f"Missing required configuration key: {'.'.join(keys)}")
            else:
                logging.debug(f"Configuration key {'.'.join(keys)} not found. Using default: {default}")
                return default
    logging.debug(f"Configuration value for {'.'.join(keys)}: {value}")
    return value