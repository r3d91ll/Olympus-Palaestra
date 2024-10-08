import random
import torch
import logging
import os
from palaestra.config import get_config_value

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    log_level = get_config_value(config, ['logging', 'level'], default='INFO')
    log_format = get_config_value(config, ['logging', 'format'], default='%(asctime)s - %(levelname)s - %(message)s')
    log_file = get_config_value(config, ['logging', 'file'], default=None)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        filename=log_file
    )

    if log_file:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)