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
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter

import deepspeed
import deepspeed.comm as dist

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def set_seed(seed):
    logging.debug("Setting seed to %d" % seed)
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
        tokenized_batch['labels'] = labels[:, 1:].contiguous()  # Shift labels by one
        tokenized_batch['input_ids'] = tokenized_batch['input_ids'][:, :-1].contiguous()  # Shift inputs
        logging.debug("Tokenization complete")
        return tokenized_batch
    return collate_fn

def load_config(config_path):
    """
    Load training configuration from a YAML file.
    """
    logging.debug("Loading configuration from %s" % config_path)
    if not os.path.exists(config_path):
        logging.error("Configuration file not found: %s" % config_path)
        sys.exit(1)
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logging.debug("Configuration loaded successfully")
        return config
    except yaml.YAMLError as e:
        logging.error("Error parsing YAML configuration file: %s" % e)
        sys.exit(1)
    except Exception as e:
        logging.error("Unexpected error loading configuration file: %s" % e)
        sys.exit(1)

def get_config_value(config, keys, default=None, required=False):
    """
    Retrieve a value from a nested configuration dictionary.
    """
    logging.debug("Retrieving config value for keys: %s" % keys)
    value = config
    for key in keys:
        if key in value:
            value = value[key]
        else:
            if required:
                logging.error("Missing required configuration key: %s" % '.'.join(keys))
                raise KeyError("Missing required configuration key: %s" % '.'.join(keys))
            else:
                logging.debug("Configuration key %s not found. Using default: %s" % ('.'.join(keys), default))
                return default
    logging.debug("Configuration value for %s: %s" % ('.'.join(keys), value))
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
        logging.debug("Dataset length: %d" % length)
        return length

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.
        """
        record = self.data[idx]
        input_text = record.get('input_text', '')
        target_text = record.get('target_text', '')
        # Combine input and target for causal language modeling
        full_text = "%s\n%s" % (input_text, target_text)
        logging.debug("Retrieved sample %d: %s" % (idx, full_text))
        return full_text

# def log_epoch_metrics(epoch_num, metrics):
#     """
#     Logs metrics for each epoch in a JSON file.
#     """
#     metrics_dir = "./logs/metrics/"
#     os.makedirs(metrics_dir, exist_ok=True)
    
#     metrics_file = "%s/metrics-epoch_%d.json" % (metrics_dir, epoch_num)
#     with open(metrics_file, 'w') as f:
#         json.dump(metrics, f, indent=4)

def calculate_metrics(logits, labels):
    """
    Calculate accuracy and F1 score for the given logits and labels.
    """
    predictions = torch.argmax(logits, dim=-1)
    # Flatten predictions and labels
    predictions = predictions.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return accuracy, f1

def validate(model_engine, val_loader, device):
    model_engine.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_f1 = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
            outputs = model_engine(**inputs)
            loss = outputs.loss.mean()
            val_loss += loss.item()
            
            accuracy, f1 = calculate_metrics(outputs.logits, inputs['labels'])
            val_accuracy += accuracy
            val_f1 += f1

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    val_f1 /= len(val_loader)
    
    return val_loss, val_accuracy, val_f1

def log_training_metrics(writer, global_step, loss, perplexity, accuracy, f1):
    writer.add_scalar('Training/Loss', loss, global_step)
    writer.add_scalar('Training/Perplexity', perplexity, global_step)
    writer.add_scalar('Training/Accuracy', accuracy, global_step)
    writer.add_scalar('Training/F1', f1, global_step)
    logging.info(f"Step {global_step}: Loss = {loss:.4f}, Perplexity = {perplexity:.2f}, Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

def log_epoch_metrics(writer, epoch, train_loss, val_loss, train_accuracy, val_accuracy, train_f1, val_f1):
    writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
    writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
    writer.add_scalar('Epoch/TrainAccuracy', train_accuracy, epoch)
    writer.add_scalar('Epoch/ValAccuracy', val_accuracy, epoch)
    writer.add_scalar('Epoch/TrainF1', train_f1, epoch)
    writer.add_scalar('Epoch/ValF1', val_f1, epoch)
    logging.info(f"Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                    f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

def save_checkpoint(model_engine, tokenizer, checkpoint_dir, epoch, global_step):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch + 1}-step{global_step}')
    os.makedirs(checkpoint_path, exist_ok=True)
    model_engine.save_checkpoint(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    logging.info(f"Checkpoint saved at step {global_step}")

def save_best_model(model_engine, tokenizer, config, epoch, global_step):
    trained_models_dir = get_config_value(config, ['paths', 'trained_models_dir'], required=True)
    model_save_dir = os.path.join(
        trained_models_dir,
        f"{config['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(model_save_dir, exist_ok=True)
    model_engine.save_checkpoint(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    logging.info(f"Best model saved at epoch {epoch + 1}, step {global_step}")

def load_deepspeed_config(config):
    deepspeed_config_path = get_config_value(config, ['paths', 'deepspeed_config'], default='config/ds_config.json')
    logging.debug(f"DeepSpeed config path: {deepspeed_config_path}")
    if not os.path.exists(deepspeed_config_path):
        logging.error(f"DeepSpeed configuration file not found: {deepspeed_config_path}")
        sys.exit(1)
    with open(deepspeed_config_path, 'r') as f:
        ds_config = json.load(f)
    logging.debug("DeepSpeed configuration loaded")
    return ds_config

def load_model_and_tokenizer(config):
    model_name = get_config_value(config, ['model', 'name'], required=True)
    models_to_train_dir = get_config_value(config, ['paths', 'models_to_train_dir'], required=True)
    model_path = os.path.join(models_to_train_dir, model_name)
    
    logging.debug(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.debug("Tokenizer loaded successfully")
    
    logging.debug(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    logging.debug("Model loaded successfully")
    
    special_tokens_dict = get_config_value(config, ['model', 'special_tokens'], default={})
    if special_tokens_dict:
        logging.debug("Adding special tokens to tokenizer")
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        logging.debug(f"Added {num_added_toks} special tokens and resized model embeddings")
        logging.info(f"Added {num_added_toks} special tokens to the tokenizer and resized model embeddings.")
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")
        logging.debug("Set tokenizer.pad_token to eos_token")
    
    logging.debug("Enabling gradient checkpointing")
    model.gradient_checkpointing_enable()
    logging.info("Enabled gradient checkpointing to reduce memory usage.")
    
    return model, tokenizer

def setup_deepspeed(args, model, ds_config):
    logging.debug("Preparing optimizer parameters")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": ds_config["optimizer"]["params"].get("weight_decay", 0.0),
        },
    ]
    logging.debug("Optimizer parameters prepared")

    logging.debug("Initializing DeepSpeed")
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters,
        config_params=ds_config
    )
    logging.debug("DeepSpeed initialized successfully")
    
    return model_engine, optimizer, _, lr_scheduler

def load_dataset(config):
    dataset_path = get_config_value(config, ['paths', 'preprocessed_dataset'], required=True)
    logging.debug(f"Loading dataset from {dataset_path}")
    if not os.path.exists(dataset_path):
        logging.error(f"Preprocessed dataset not found: {dataset_path}")
        sys.exit(1)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    logging.debug(f"Loaded dataset with {len(data)} samples")
    logging.info(f"Loaded dataset with {len(data)} samples.")
    return data

def setup_tensorboard(config):
    log_dir = get_config_value(config, ['paths', 'logs_dir'], required=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logging set to {log_dir}")
    logging.debug("TensorBoard SummaryWriter initialized")
    return writer

def setup_checkpointing(config):
    checkpoint_dir = get_config_value(config, ['paths', 'checkpoints_dir'], required=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.info(f"Checkpoints will be saved to {checkpoint_dir}")
    logging.debug("Checkpoint directory created")
    return checkpoint_dir

def get_training_params(config):
    batch_size = get_config_value(config, ['training', 'batch_size'], default=12)
    validation_batch_size = get_config_value(config, ['training', 'validation_batch_size'], default=1)
    num_epochs = get_config_value(config, ['training', 'epochs'], default=10)
    num_workers = get_config_value(config, ['training', 'num_workers'], default=4)
    checkpoint_interval = get_config_value(config, ['training', 'checkpoint_interval'], default=1000)
    logging_interval = get_config_value(config, ['training', 'logging_interval'], default=100)
    max_length = get_config_value(config, ['training', 'max_sequence_length'], default=1024)
    warmup_steps = get_config_value(config, ['training', 'warmup_steps'], default=0)
    
    return (batch_size, validation_batch_size, num_epochs, num_workers, 
            checkpoint_interval, logging_interval, max_length, warmup_steps)

def main():
    logging.info("Starting main function")
    parser = argparse.ArgumentParser(description='Palaestra Training Script')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank passed from distributed launcher')
    parser.add_argument('--config', type=str, help='Path to the config.yaml file.')
    parser = deepspeed.add_config_arguments(parser)
    args, unknown = parser.parse_known_args()

    logging.debug("Parsed arguments: %s" % args)
    logging.debug("Unknown arguments: %s" % unknown)

    # Load configuration and set up model
    config = load_config(args.config)
    ds_config = load_deepspeed_config(config)
    model, tokenizer = load_model_and_tokenizer(config)

    # Set up DeepSpeed
    model_engine, optimizer, _, lr_scheduler = setup_deepspeed(args, model, ds_config)
    rank = model_engine.local_rank
    world_size = model_engine.world_size
    device = model_engine.device

    # Set seed for reproducibility
    set_seed(42)

    # Load and split dataset
    data = load_dataset(config)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    logging.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Set up TensorBoard logging
    writer = setup_tensorboard(config) if rank == 0 else None

    # Set up checkpointing
    checkpoint_dir = setup_checkpointing(config)

    # Training parameters
    training_params = get_training_params(config)
    batch_size, validation_batch_size, num_epochs, num_workers, checkpoint_interval, logging_interval, max_length, warmup_steps = training_params

    # Prepare datasets and data loaders
    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    collate_fn = get_collate_fn(tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=validation_batch_size, sampler=val_sampler,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_f1 = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") if rank == 0 else train_loader

        for step, batch in enumerate(train_loader_tqdm):
            inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
            
            outputs = model_engine(**inputs)
            loss = outputs.loss
            loss = loss.mean()
            
            model_engine.backward(loss)
            model_engine.step()
            
            if lr_scheduler:
                lr_scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Calculate additional metrics
            with torch.no_grad():
                accuracy, f1 = calculate_metrics(outputs.logits, inputs['labels'])
                epoch_accuracy += accuracy
                epoch_f1 += f1

            if global_step % logging_interval == 0 and rank == 0:
                avg_loss = epoch_loss / (step + 1)
                avg_accuracy = epoch_accuracy / (step + 1)
                avg_f1 = epoch_f1 / (step + 1)
                perplexity = math.exp(avg_loss)
                log_training_metrics(writer, global_step, avg_loss, perplexity, avg_accuracy, avg_f1)

            if global_step % checkpoint_interval == 0:
                save_checkpoint(model_engine, tokenizer, checkpoint_dir, epoch, global_step)

        # Validation
        val_loss, val_accuracy, val_f1 = validate(model_engine, val_loader, device)
        
        if rank == 0:
            avg_train_loss = epoch_loss / len(train_loader)
            avg_train_accuracy = epoch_accuracy / len(train_loader)
            avg_train_f1 = epoch_f1 / len(train_loader)
            log_epoch_metrics(writer, epoch, avg_train_loss, val_loss, avg_train_accuracy, val_accuracy, avg_train_f1, val_f1)
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_best_model(model_engine, tokenizer, config, epoch, global_step)

    if rank == 0:
        writer.close()

if __name__ == '__main__':
    logging.debug("Executing main function")
    main()
    logging.debug("Main function execution completed")