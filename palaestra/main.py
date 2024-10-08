import argparse
import logging
from palaestra.config import config
from palaestra.utils.huggingface.huggingface_utils import ensure_model_and_dataset
from palaestra.askesis.deepspeed_regime import deepspeed_training_regime

def main():
    parser = argparse.ArgumentParser(description='Palaestra Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()

    # Load configuration
    cfg = config.load_config(args.config)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ensure model and dataset are available
    ensure_model_and_dataset()

    # Start training
    deepspeed_training_regime(cfg, args)

if __name__ == '__main__':
    main()