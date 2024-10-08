import argparse
import logging
from palaestra.utils.deepspeed.utils import (
    initialize_deepspeed,
    get_optimizer_params,
    load_ds_config,
    deepspeed_train_step,
    deepspeed_eval_step,
    save_deepspeed_checkpoint
)
from palaestra.data.dataset import TextDataset, create_dataloaders
from palaestra.models.model_loader import load_model_and_tokenizer
from palaestra.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_llama_function_calling(args):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.get('model', 'name'), config.get('paths', 'models_dir'))

    # Prepare dataset
    dataset = TextDataset(
        data=config.get('data', 'function_calling_dataset'),
        tokenizer=tokenizer,
        max_length=config.get('training', 'max_sequence_length')
    )
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=config.get('training', 'batch_size'),
        num_workers=config.get('training', 'num_workers')
    )

    # Initialize DeepSpeed
    ds_config = load_ds_config(config.get('paths', 'deepspeed_config'))
    optimizer_params = get_optimizer_params(model, ds_config['optimizer']['params']['weight_decay'])
    model_engine, _ = initialize_deepspeed(model, optimizer_params, args, ds_config)

    # Training loop
    for epoch in range(config.get('training', 'epochs')):
        logger.info(f"Starting epoch {epoch + 1}")
        train_loss = 0
        for batch in train_loader:
            loss = deepspeed_train_step(model_engine, batch)
            train_loss += loss

        # Evaluation
        val_loss = 0
        for batch in val_loader:
            val_loss += deepspeed_eval_step(model_engine, batch)

        logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, "
                    f"Val Loss: {val_loss / len(val_loader)}")

        # Save checkpoint
        if (epoch + 1) % config.get('training', 'save_every') == 0:
            save_deepspeed_checkpoint(model_engine, f"{config.get('paths', 'checkpoints_dir')}/epoch_{epoch + 1}")

    # Save final model
    save_deepspeed_checkpoint(model_engine, f"{config.get('paths', 'trained_models_dir')}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLaMA model for function calling")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    train_llama_function_calling(args)