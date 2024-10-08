import logging
from torch.utils.tensorboard import SummaryWriter
from palaestra.config import config
import argparse
import os
import sys
import json
from datetime import datetime
from tqdm import tqdm
import math
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)

# Import DeepSpeed
import deepspeed
import deepspeed.comm as dist

# Import Palaestra modules
from palaestra.data import TextDataset, get_collate_fn
from palaestra.utils.huggingface_utils import ensure_model_and_dataset, load_model_and_tokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def set_seed(seed):
    logging.debug(f"Setting seed to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_epoch_metrics(epoch_num, metrics):
    metrics_dir = "./logs/metrics/"
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = f"{metrics_dir}/metrics-epoch_{epoch_num}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    logging.info("Starting main function")
    parser = argparse.ArgumentParser(description='Palaestra Training Script')

    # Add --local_rank argument to handle DeepSpeed's injected argument
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank passed from distributed launcher')

    # DeepSpeed adds its own command-line arguments, so we need to pass the parser to DeepSpeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    logging.debug(f"Parsed arguments: {args}")

    # Ensure model and dataset are available
    ensure_model_and_dataset()

    # Load DeepSpeed configuration
    deepspeed_config_path = config.get('paths', 'deepspeed_config', default='config/ds_config.json')
    logging.debug(f"DeepSpeed config path: {deepspeed_config_path}")
    if not os.path.exists(deepspeed_config_path):
        logging.error(f"DeepSpeed configuration file not found: {deepspeed_config_path}")
        sys.exit(1)
    with open(deepspeed_config_path, 'r') as f:
        ds_config = json.load(f)
    logging.debug("DeepSpeed configuration loaded")

    # Load model and tokenizer
    model_name = config.get('model', 'name')
    models_dir = config.get('paths', 'models_to_train_dir')
    model_path = os.path.join(models_dir, model_name)
    model, tokenizer = load_model_and_tokenizer(model_name, model_path)

    # Update tokenizer with special tokens if specified
    special_tokens_dict = config.get('model', 'special_tokens', default={})
    if special_tokens_dict:
        logging.debug("Adding special tokens to tokenizer")
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        logging.info(f"Added {num_added_toks} special tokens to the tokenizer and resized model embeddings.")
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")

    # Enable gradient checkpointing
    logging.debug("Enabling gradient checkpointing")
    model.gradient_checkpointing_enable()
    logging.info("Enabled gradient checkpointing to reduce memory usage.")

    # Prepare optimizer parameters
    logging.debug("Preparing optimizer parameters")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": ds_config["optimizer"]["params"].get("weight_decay", 0.0),
        },
    ]

    # Initialize DeepSpeed
    logging.debug("Initializing DeepSpeed")
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters,
        config_params=ds_config
    )
    logging.debug("DeepSpeed initialized successfully")

    # Set up device, rank, and world size
    device = model_engine.device
    rank = model_engine.local_rank
    world_size = model_engine.world_size
    logging.debug(f"Device: {device}, Local rank: {rank}, World size: {world_size}")

    if rank == 0:
        logging.info(f"Using device: {device}")
        logging.info(f"Running on {world_size} processes")

    # Set seed for reproducibility
    seed = config.get('training', 'seed', default=42)
    set_seed(seed)

    # Load and prepare dataset
    dataset_path = config.get('paths', 'preprocessed_dataset')
    if not os.path.exists(dataset_path):
        logging.error(f"Preprocessed dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    logging.info(f"Loaded dataset with {len(data)} samples.")

    # Split dataset into training and validation sets
    validation_split = config.get('training', 'validation_split', default=0.1)
    split_index = int(len(data) * (1 - validation_split))
    train_data = data[:split_index]
    val_data = data[split_index:]

    logging.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Prepare datasets and dataloaders
    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)

    batch_size = config.get('training', 'batch_size', default=4)
    validation_batch_size = config.get('training', 'validation_batch_size', default=1)
    num_workers = config.get('training', 'num_workers', default=4)
    max_length = config.get('training', 'max_sequence_length', default=1024)

    collate_fn = get_collate_fn(tokenizer, max_length)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Set up TensorBoard logging
    if rank == 0:
        log_dir = config.get('paths', 'logs_dir', default='/app/logs')
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"TensorBoard logging set to {log_dir}")

    # Set up checkpointing
    checkpoint_dir = config.get('paths', 'checkpoints_dir')
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoints will be saved to {checkpoint_dir}")

    # Training loop setup
    num_epochs = config.get('training', 'epochs', default=3)
    checkpoint_interval = config.get('training', 'checkpoint_interval', default=1000)
    logging_interval = config.get('training', 'logging_interval', default=100)

    global_step = 0
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_start_time = datetime.now()

        train_sampler.set_epoch(epoch)
        model_engine.train()
        epoch_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") if rank == 0 else train_loader

        for step, batch in enumerate(train_loader_tqdm):
            inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
            outputs = model_engine(**inputs)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % logging_interval == 0 and rank == 0:
                avg_loss = epoch_loss / (step + 1)
                writer.add_scalar('Training/Loss', avg_loss, global_step)
                perplexity = math.exp(avg_loss)
                writer.add_scalar('Training/Perplexity', perplexity, global_step)

            if global_step % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch + 1}-step{global_step}')
                model_engine.save_checkpoint(checkpoint_path)
                if rank == 0:
                    tokenizer.save_pretrained(checkpoint_path)
                    logging.info(f"Checkpoint saved at step {global_step}")

        # Validation
        model_engine.eval()
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Validation") if rank == 0 else val_loader

        with torch.no_grad():
            for batch in val_loader_tqdm:
                inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
                outputs = model_engine(**inputs)
                loss = outputs.loss
                val_loss += loss.item()

        val_loss = torch.tensor(val_loss).to(device)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss = val_loss.item() / len(val_loader) / world_size

        if rank == 0:
            avg_train_loss = epoch_loss / len(train_loader)
            perplexity = math.exp(val_loss)
            writer.add_scalar('Validation/Loss', val_loss, epoch + 1)
            writer.add_scalar('Validation/Perplexity', perplexity, epoch + 1)

            logging.info(f"Epoch {epoch + 1} completed. Avg Training Loss: {avg_train_loss:.4f}, "
                         f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

            # Log metrics to JSON file
            epoch_duration = datetime.now() - epoch_start_time
            metrics = {
                "epoch": epoch + 1,
                "training_loss": avg_train_loss,
                "validation_loss": val_loss,
                "perplexity": perplexity,
                "epoch_duration": str(epoch_duration)
            }
            log_epoch_metrics(epoch + 1, metrics)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank == 0:
                best_model_path = os.path.join(config.get('paths', 'trained_models_dir'), 'best_model')
                model_engine.save_checkpoint(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

    if rank == 0:
        writer.close()
    logging.info("Training completed")

if __name__ == '__main__':
    logging.debug("Executing main function")
    main()
    logging.debug("Main function execution completed")