import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self._config = {}
        self.load_config()

    def load_config(self):
        logging.info(f"Loading configuration from {self.config_path}")
        if not os.path.exists(self.config_path):
            logging.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as config_file:
                self._config = json.load(config_file)
            logging.info("Configuration loaded successfully")
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading configuration: {e}")
            raise

    def get(self, *keys, default=None):
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                logging.warning(f"Configuration key not found: {'.'.join(map(str, keys))}")
                return default
        return value

    def set(self, value, *keys):
        if not keys:
            logging.error("No keys provided to set configuration value")
            return

        config = self._config
        for key in keys[:-1]:
            config = config.setdefault(key, {})

        config[keys[-1]] = value
        logging.info(f"Configuration updated: {'.'.join(map(str, keys))} = {value}")

    def save_config(self):
        logging.info(f"Saving configuration to {self.config_path}")
        try:
            with open(self.config_path, 'w') as config_file:
                json.dump(self._config, config_file, indent=4)
            logging.info("Configuration saved successfully")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            raise

# Create a global config object
config = Config(os.environ.get('PALAESTRA_CONFIG', 'config/settings.json'))