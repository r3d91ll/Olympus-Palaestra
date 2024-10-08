import deepspeed
import torch
from palaestra.training.trainer import DeepSpeedTrainer
from palaestra.models.model_loader import load_model_and_tokenizer
from palaestra.data.dataset import create_dataloaders, TextDataset
from palaestra.evaluation.evaluator import evaluate_model
from palaestra.logging.logger import setup_logging

def deepspeed_training_regime(config, args):
    logger = setup_logging(config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config['model_name'], config['model_path'])
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(config['train_data'], tokenizer, config['max_length'])
    val_dataset = TextDataset(config['val_data'], tokenizer, config['max_length'])
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config['batch_size'], config['num_workers'])
    
    # Initialize DeepSpeed trainer
    trainer = DeepSpeedTrainer(model, train_loader, val_loader, config, args)
    
    # Train the model
    for epoch in range(config['num_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
        t