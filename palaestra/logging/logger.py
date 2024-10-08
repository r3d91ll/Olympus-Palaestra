import logging
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]):
    logging.basicConfig(
        level=config.get('log_level', logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=config.get('log_file', None)
    )
    return logging.getLogger(__name__)