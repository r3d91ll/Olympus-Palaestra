#!/usr/bin/env python3
# palaestra.py

"""
Palaestra Training Script for Palaestra's Gymnasia

This script trains a language model using preprocessed data.
It loads configurations from 'config.yaml', prepares the data,
sets up the model, and runs the training loop with logging and checkpointing.
It uses DeepSpeed for efficient multi-GPU training with ZeRO-Offload.

Usage:
    deepspeed scripts/palaestra.py --config config/config.yaml
"""

import argparse
import os
import sys
import json
import yaml
import logging
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
)
from torch.utils.tensorboard import SummaryWriter

# Import DeepSpeed
import deepspeed
import deepspeed.comm as dist  # Alias deepspeed.comm as dist

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for logging key events
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def set_seed(seed):
    logging.debug(f"Setting seed to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_collate_fn(tokenizer, max_length):
    """
    Returns a collate function that tokenizes input texts for the DataLoader.
    """
    logging.debug("Creating collate function")
    def collate_fn(batch_texts):
        logging.debug("Tokenizing batch texts")
        tokenized_batch = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding='longest',
            return_tensors='pt'
        )
        labels = tokenized_batch['input_ids'].clone()
        tokenized_batch['labels'] = labels
        logging.debug("Tokenization complete")
        return tokenized_batch
    return collate_fn


def load_config(config_path):
    """
    Load training configuration from a YAML file.
    """
    logging.debug(f"Loading configuration from {config_path}")
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
    """
    Retrieve a value from a nested configuration dictionary.
    """
    logging.debug(f"Retrieving config value for keys: {keys}")
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


class TextDataset(Dataset):
    """
    Custom Dataset for loading preprocessed text data.
    """
    def __init__(self, data):
        logging.debug("Initializing TextDataset")
        self.data = data

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        length = len(self.data)
        logging.debug(f"Dataset length: {length}")
        return length

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.
        """
        record = self.data[idx]
        input_text = record.get('input_text', '')
        target_text = record.get('target_text', '')
        # Combine input and target for causal language modeling
        full_text = f"{input_text}\n{target_text}"
        logging.debug(f"Retrieved sample {idx}: {full_text}")
        return full_text


def log_epoch_metrics(epoch_num, metrics):
    """
    Logs metrics for each epoch in a JSON file.
    """
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
    parser.add_argument('--config', type=str, help='Path to the config.yaml file.')
    parser = deepspeed.add_config_arguments(parser)
    args, unknown = parser.parse_known_args()

    logging.debug(f"Parsed arguments: {args}")
    logging.debug(f"Unknown arguments: {unknown}")

    # Initialize DeepSpeed first to set up communication backend
    try:
        # Load configuration early to access DeepSpeed config path
        if not args.config:
            logging.error("Configuration file is required. Use --config to specify the path.")
            sys.exit(1)
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

    try:
        # Load DeepSpeed configuration
        deepspeed_config_path = get_config_value(config, ['paths', 'deepspeed_config'], default='config/ds_config.json')
        logging.debug(f"DeepSpeed config path: {deepspeed_config_path}")
        if not os.path.exists(deepspeed_config_path):
            logging.error(f"DeepSpeed configuration file not found: {deepspeed_config_path}")
            sys.exit(1)
        with open(deepspeed_config_path, 'r') as f:
            ds_config = json.load(f)
        logging.debug("DeepSpeed configuration loaded")
    except Exception as e:
        logging.error(f"Error loading DeepSpeed configuration: {e}")
        sys.exit(1)

    try:
        # Load tokenizer and model
        model_name = get_config_value(config, ['model', 'name'], required=True)
        models_to_train_dir = get_config_value(config, ['paths', 'models_to_train_dir'], required=True)
        model_path = os.path.join(models_to_train_dir, model_name)
        logging.debug(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.debug("Tokenizer loaded successfully")
        logging.debug(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        logging.debug("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

    try:
        # Update tokenizer with special tokens if specified
        special_tokens_dict = get_config_value(config, ['model', 'special_tokens'], default={})
        if special_tokens_dict:
            logging.debug("Adding special tokens to tokenizer")
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))
            logging.debug(f"Added {num_added_toks} special tokens and resized model embeddings")
            logging.info(f"Added {num_added_toks} special tokens to the tokenizer and resized model embeddings.")
        else:
            # If pad_token is not set, set it to eos_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logging.info(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")
                logging.debug("Set tokenizer.pad_token to eos_token")
    except Exception as e:
        logging.error(f"Error updating tokenizer with special tokens: {e}")
        sys.exit(1)

    try:
        # Enable gradient checkpointing
        logging.debug("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing to reduce memory usage.")
    except Exception as e:
        logging.error(f"Error enabling gradient checkpointing: {e}")
        sys.exit(1)

    try:
        # Prepare optimizer parameters
        logging.debug("Preparing optimizer parameters")
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": ds_config["optimizer"]["params"].get("weight_decay", 0.0),
            },
        ]
        logging.debug("Optimizer parameters prepared")
    except Exception as e:
        logging.error(f"Error preparing optimizer parameters: {e}")
        sys.exit(1)

    try:
        # Initialize DeepSpeed
        logging.debug("Initializing DeepSpeed")
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=optimizer_grouped_parameters,
            config_params=ds_config
        )
        logging.debug("DeepSpeed initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing DeepSpeed: {e}")
        sys.exit(1)

    try:
        # Now, access rank and world_size via model_engine
        rank = model_engine.local_rank
        world_size = model_engine.world_size
        logging.debug(f"Local rank: {rank}, World size: {world_size}")
        if rank == 0:
            logging.info(f"Using device: {model_engine.device}")
            logging.info(f"Running on {world_size} processes")
    except Exception as e:
        logging.error(f"Error accessing rank and world size: {e}")
        sys.exit(1)

    try:
        # Set seed for reproducibility
        seed = 42  # Or any number you prefer
        set_seed(seed)
    except Exception as e:
        logging.error(f"Error setting seed: {e}")
        sys.exit(1)

    try:
        # Set device
        device = model_engine.device
        logging.debug(f"Device set to: {device}")
    except Exception as e:
        logging.error(f"Error setting device: {e}")
        sys.exit(1)

    try:
        # Load dataset
        dataset_path = get_config_value(config, ['paths', 'preprocessed_dataset'], required=True)
        logging.debug(f"Loading dataset from {dataset_path}")
        if not os.path.exists(dataset_path):
            if rank == 0:
                logging.error(f"Preprocessed dataset not found: {dataset_path}")
            sys.exit(1)
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        logging.debug(f"Loaded dataset with {len(data)} samples")
        if rank == 0:
            logging.info(f"Loaded dataset with {len(data)} samples.")
    except Exception as e:
        if rank == 0:
            logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

    try:
        # Shuffle data before splitting
        logging.debug("Shuffling dataset")
        random.shuffle(data)
    except Exception as e:
        logging.error(f"Error shuffling dataset: {e}")
        sys.exit(1)

    try:
        # Split dataset into training and validation sets
        validation_split = get_config_value(config, ['training', 'validation_split'], default=0.1)
        split_index = int(len(data) * (1 - validation_split))
        train_data = data[:split_index]
        val_data = data[split_index:]
        logging.debug(f"Split dataset into {len(train_data)} training and {len(val_data)} validation samples")
    except Exception as e:
        logging.error(f"Error splitting dataset: {e}")
        sys.exit(1)

    if rank == 0:
        logging.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    try:
        # Prepare datasets
        logging.debug("Preparing training and validation datasets")
        train_dataset = TextDataset(train_data)
        val_dataset = TextDataset(val_data)
        logging.debug("Datasets prepared")
    except Exception as e:
        logging.error(f"Error preparing datasets: {e}")
        sys.exit(1)

    try:
        # Retrieve training parameters
        batch_size = get_config_value(config, ['training', 'batch_size'], default=4)
        validation_batch_size = get_config_value(config, ['training', 'validation_batch_size'], default=1)
        num_epochs = get_config_value(config, ['training', 'epochs'], default=3)
        num_workers = get_config_value(config, ['training', 'num_workers'], default=4)
        checkpoint_interval = get_config_value(config, ['training', 'checkpoint_interval'], default=1000)
        logging_interval = get_config_value(config, ['training', 'logging_interval'], default=100)
        use_amp = get_config_value(config, ['training', 'use_amp'], default=True)
        max_length = get_config_value(config, ['training', 'max_sequence_length'], default=1024)

        logging.debug(f"Training parameters: batch_size={batch_size}, validation_batch_size={validation_batch_size}, "
                      f"num_epochs={num_epochs}, num_workers={num_workers}, checkpoint_interval={checkpoint_interval}, "
                      f"logging_interval={logging_interval}, use_amp={use_amp}, max_length={max_length}")
    except Exception as e:
        logging.error(f"Error retrieving training parameters: {e}")
        sys.exit(1)

    try:
        # Create collate function
        collate_fn = get_collate_fn(tokenizer, max_length)
    except Exception as e:
        logging.error(f"Error creating collate function: {e}")
        sys.exit(1)

    try:
        # Create DistributedSampler for training and validation datasets
        logging.debug("Creating DistributedSamplers")
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        logging.debug("DistributedSamplers created")
    except Exception as e:
        logging.error(f"Error creating DistributedSamplers: {e}")
        sys.exit(1)

    try:
        # Create DataLoaders with the DistributedSampler
        logging.debug("Creating DataLoaders")
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
        logging.debug("DataLoaders created")
    except Exception as e:
        logging.error(f"Error creating DataLoaders: {e}")
        sys.exit(1)

    try:
        # Set up TensorBoard logging
        if rank == 0:
            try:
                log_dir = get_config_value(config, ['paths', 'logs_dir'], required=True)
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                logging.info(f"TensorBoard logging set to {log_dir}")
                logging.debug("TensorBoard SummaryWriter initialized")
            except Exception as e:
                logging.error(f"Error setting up TensorBoard: {e}")
                sys.exit(1)
    except Exception as e:
        logging.error(f"Error in TensorBoard setup: {e}")
        sys.exit(1)

    try:
        # Set up checkpointing
        checkpoint_dir = get_config_value(config, ['paths', 'checkpoints_dir'], required=True)
        if rank == 0:
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                logging.info(f"Checkpoints will be saved to {checkpoint_dir}")
                logging.debug("Checkpoint directory created")
            except Exception as e:
                logging.error(f"Error creating checkpoint directory: {e}")
                sys.exit(1)
    except Exception as e:
        logging.error(f"Error in checkpoint setup: {e}")
        sys.exit(1)

    # Initialize variables for tracking
    global_step = 0
    best_val_loss = float('inf')
    logging.debug(f"Initial best_val_loss: {best_val_loss}")

    # Training loop
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_start_time = datetime.now()

        try:
            train_sampler.set_epoch(epoch)
            model_engine.train()
            epoch_loss = 0.0
        except Exception as e:
            logging.error(f"Error during epoch setup: {e}")
            sys.exit(1)

        # Create a progress bar only on the main process
        if rank == 0:
            train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            logging.debug("Progress bar for training initialized")
        else:
            train_loader_tqdm = train_loader

        for step, batch in enumerate(train_loader_tqdm):
            logging.debug(f"Epoch {epoch + 1}, Step {step + 1}: Starting training step")
            try:
                inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
            except Exception as e:
                logging.error(f"Error moving batch to device: {e}")
                sys.exit(1)

            try:
                outputs = model_engine(**inputs)
                loss = outputs.loss
                loss = loss.mean()
                logging.debug(f"Epoch {epoch + 1}, Step {step + 1}: Loss computed: {loss.item()}")
                model_engine.backward(loss)
                model_engine.step()
                logging.debug(f"Epoch {epoch + 1}, Step {step + 1}: Backward and step completed")
            except Exception as e:
                if rank == 0:
                    logging.error(f"Error during training step {step + 1}: {e}")
                sys.exit(1)

            epoch_loss += loss.item()
            global_step += 1
            logging.debug(f"Epoch {epoch + 1}, Step {step + 1}: Global step updated to {global_step}")

            # Log training metrics only on the main process
            if global_step % logging_interval == 0 and rank == 0:
                avg_loss = epoch_loss / (step + 1)
                try:
                    writer.add_scalar('Training/Loss', avg_loss, global_step)
                    perplexity = math.exp(avg_loss)
                    writer.add_scalar('Training/Perplexity', perplexity, global_step)
                    logging.info(f"Step {global_step}: Avg Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")
                    logging.debug(f"Step {global_step}: Training metrics logged")
                except Exception as e:
                    logging.error(f"Error logging training metrics at step {global_step}: {e}")

            # Save checkpoint on all processes
            if global_step % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch + 1}-step{global_step}')
                if rank == 0:
                    try:
                        os.makedirs(checkpoint_path, exist_ok=True)
                        logging.info(f"Saving checkpoint to {checkpoint_path}")
                        logging.debug("Checkpoint directory created for saving")
                    except Exception as e:
                        logging.error(f"Error creating checkpoint directory {checkpoint_path}: {e}")
                        sys.exit(1)
                dist.barrier()  # Synchronize before saving checkpoint

                # All ranks call save_checkpoint()
                try:
                    model_engine.save_checkpoint(checkpoint_path)
                    logging.debug(f"Checkpoint saved at {checkpoint_path}")
                except Exception as e:
                    if rank == 0:
                        logging.error(f"Error saving checkpoint at step {global_step}: {e}")
                    sys.exit(1)

                # Only Rank 0 saves the tokenizer and logs
                if rank == 0:
                    try:
                        tokenizer.save_pretrained(checkpoint_path)
                        logging.info(f"Checkpoint and tokenizer saved at step {global_step}")
                        logging.debug("Tokenizer saved successfully")
                    except Exception as e:
                        logging.error(f"Error saving tokenizer at step {global_step}: {e}")

                # Synchronize after checkpoint saving
                dist.barrier()
                logging.debug("Checkpoint saving synchronized across all ranks")

        # Validation
        logging.info(f"Epoch {epoch + 1}: Starting validation")
        try:
            model_engine.eval()
            model_engine.module.eval()  # Ensure the underlying model is in eval mode
        except Exception as e:
            logging.error(f"Error setting model to eval mode: {e}")
            sys.exit(1)

        try:
            # Disable gradient checkpointing for validation
            model_engine.module.gradient_checkpointing_disable()
            logging.debug("Gradient checkpointing disabled for validation")

            # Clear CUDA cache to free up memory
            torch.cuda.empty_cache()
            logging.debug("CUDA cache cleared")
        except Exception as e:
            logging.error(f"Error during validation setup: {e}")
            sys.exit(1)

        val_loss = 0.0

        # Create a progress bar only on the main process
        if rank == 0:
            val_loader_tqdm = tqdm(val_loader, desc="Validation")
            logging.debug("Progress bar for validation initialized")
        else:
            val_loader_tqdm = val_loader

        with torch.no_grad():
            for batch in val_loader_tqdm:
                logging.debug(f"Epoch {epoch + 1}: Starting validation step")
                try:
                    inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
                except Exception as e:
                    logging.error(f"Error moving validation batch to device: {e}")
                    sys.exit(1)

                try:
                    outputs = model_engine.module(**inputs)  # Use the underlying model
                    loss = outputs.loss
                    loss = loss.mean()
                    logging.debug(f"Epoch {epoch + 1}: Validation loss computed: {loss.item()}")
                except Exception as e:
                    if rank == 0:
                        logging.error(f"Error during validation step: {e}")
                    sys.exit(1)
                val_loss += loss.item()
                logging.debug(f"Epoch {epoch + 1}: Validation loss accumulated: {val_loss}")

        try:
            # Aggregate validation loss across processes
            val_loss_tensor = torch.tensor(val_loss).to(device)
            logging.debug(f"Aggregating validation loss: {val_loss_tensor.item()}")
            dist.reduce(val_loss_tensor, op=dist.ReduceOp.SUM, dst=0)
            logging.debug("Validation loss aggregated")
        except Exception as e:
            logging.error(f"Error aggregating validation loss: {e}")
            sys.exit(1)

        if rank == 0:
            try:
                avg_val_loss = val_loss_tensor.item() / world_size / len(val_loader)
                avg_train_loss = epoch_loss / len(train_loader)
                perplexity = math.exp(avg_val_loss)
                logging.info(f"Epoch {epoch + 1} completed. Avg Training Loss: {avg_train_loss:.4f}, "
                                f"Avg Validation Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")
                logging.debug(f"Epoch {epoch + 1}: Metrics calculated")

                # Log validation metrics
                writer.add_scalar('Validation/Loss', avg_val_loss, epoch + 1)
                writer.add_scalar('Validation/Perplexity', perplexity, epoch + 1)
                logging.debug("Validation metrics logged")

                # Log metrics to JSON file
                epoch_duration = datetime.now() - epoch_start_time
                metrics = {
                    "epoch": epoch + 1,
                    "training_loss": avg_train_loss,
                    "validation_loss": avg_val_loss,
                    "perplexity": perplexity,
                    "epoch_duration": str(epoch_duration)
                }
                log_epoch_metrics(epoch + 1, metrics)
                logging.info(f"Epoch metrics logged to JSON for epoch {epoch + 1}")
                
            except Exception as e:
                logging.error(f"Error calculating or logging validation metrics: {e}")
                sys.exit(1)

        try:
            # Synchronize before proceeding
            dist.barrier()
            logging.debug("Synchronized after validation")
        except Exception as e:
            logging.error(f"Error during synchronization: {e}")
            sys.exit(1)

        # Save the best model on all ranks
        if rank == 0:
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                logging.info(f"New best validation loss: {best_val_loss:.4f}")
                logging.debug("Best validation loss updated")
        else:
            is_best = False  # Default value on other ranks

        try:
            # Convert is_best to tensor
            is_best_tensor = torch.tensor(int(is_best), device=device)
            logging.debug(f"Broadcasting is_best_tensor: {is_best_tensor.item()}")

            # Broadcast is_best_tensor to all ranks
            dist.broadcast(is_best_tensor, src=0)
            logging.debug("is_best_tensor broadcasted to all ranks")

            # Convert back to boolean
            is_best = bool(is_best_tensor.item())
            logging.debug(f"is_best after broadcasting: {is_best}")
        except Exception as e:
            logging.error(f"Error broadcasting is_best_tensor: {e}")
            sys.exit(1)

        if is_best:
            try:
                trained_models_dir = get_config_value(config, ['paths', 'trained_models_dir'], required=True)
                model_save_dir = os.path.join(
                    trained_models_dir,
                    f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                logging.debug(f"Preparing to save the best model to {model_save_dir}")
                if rank == 0:
                    os.makedirs(model_save_dir, exist_ok=True)
                    logging.info(f"Saving the best model to {model_save_dir}")
            except Exception as e:
                logging.error(f"Error creating best model directory {model_save_dir}: {e}")
                sys.exit(1)

            try:
                dist.barrier()  # Ensure directory is created

                # All ranks save the checkpoint
                model_engine.save_checkpoint(model_save_dir)
                logging.debug(f"Best checkpoint saved to {model_save_dir}")
            except Exception as e:
                if rank == 0:
                    logging.error(f"Error saving best checkpoint: {e}")
                sys.exit(1)

            if rank == 0:
                try:
                    tokenizer.save_pretrained(model_save_dir)
                    logging.info(f"Best model and tokenizer saved to {model_save_dir}")
                    logging.debug("Tokenizer saved successfully for the best model")
                except Exception as e:
                    logging.error(f"Error saving tokenizer for best model: {e}")

            try:
                # Synchronize after saving
                dist.barrier()
                logging.debug("Synchronized after saving the best model")
            except Exception as e:
                logging.error(f"Error during synchronization after saving best model: {e}")
                sys.exit(1)

        try:
            # Re-enable gradient checkpointing after validation
            model_engine.module.gradient_checkpointing_enable()
            logging.debug("Gradient checkpointing re-enabled after validation")
            model_engine.train()
        except Exception as e:
            logging.error(f"Error re-enabling gradient checkpointing: {e}")
            sys.exit(1)


if __name__ == '__main__':
    logging.debug("Executing main function")
    main()
    logging.debug("Main function execution completed")
