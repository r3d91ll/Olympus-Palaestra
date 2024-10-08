import os
import logging
import math
import json
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import deepspeed.comm as dist
from palaestra.config import get_config_value

def log_epoch_metrics(epoch_num, metrics):
    metrics_dir = "./logs/metrics/"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = f"{metrics_dir}/metrics-epoch_{epoch_num}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def train_model(config, model_engine, optimizer, lr_scheduler, train_loader, val_loader, tokenizer, rank):
    device = model_engine.device
    num_epochs = get_config_value(config, ['training', 'epochs'], default=3)
    checkpoint_interval = get_config_value(config, ['training', 'checkpoint_interval'], default=1000)
    logging_interval = get_config_value(config, ['training', 'logging_interval'], default=100)
    checkpoint_dir = get_config_value(config, ['paths', 'checkpoints_dir'], required=True)

    if rank == 0:
        log_dir = get_config_value(config, ['paths', 'logs_dir'], required=True)
        writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)
        model_engine.train()
        epoch_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") if rank == 0 else train_loader

        for step, batch in enumerate(train_loader_tqdm):
            inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
            outputs = model_engine(**inputs)
            loss = outputs.loss.mean()
            model_engine.backward(loss)
            model_engine.step()

            if lr_scheduler:
                lr_scheduler.step()
                if rank == 0 and global_step % logging_interval == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    writer.add_scalar('Training/Learning_Rate', current_lr, global_step)

            epoch_loss += loss.item()
            global_step += 1

            if global_step % logging_interval == 0 and rank == 0:
                avg_loss = epoch_loss / (step + 1)
                writer.add_scalar('Training/Loss', avg_loss, global_step)
                perplexity = math.exp(avg_loss)
                writer.add_scalar('Training/Perplexity', perplexity, global_step)

            if global_step % checkpoint_interval == 0:
                save_checkpoint(model_engine, tokenizer, checkpoint_dir, epoch, global_step, rank)

        val_loss = validate_model(model_engine, val_loader, device, rank)

        if rank == 0:
            avg_val_loss = val_loss / len(val_loader)
            avg_train_loss = epoch_loss / len(train_loader)
            perplexity = math.exp(avg_val_loss)
            log_validation_metrics(writer, avg_train_loss, avg_val_loss, perplexity, epoch)
            log_epoch_summary(epoch, avg_train_loss, avg_val_loss, perplexity, epoch_start_time)

        is_best = handle_best_model(avg_val_loss, best_val_loss, model_engine, tokenizer, config, rank)
        if is_best:
            best_val_loss = avg_val_loss

        dist.barrier()

def validate_model(model_engine, val_loader, device, rank):
    model_engine.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc="Validation") if rank == 0 else val_loader

    with torch.no_grad():
        for batch in val_loader_tqdm:
            inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
            outputs = model_engine(**inputs)
            loss = outputs.loss.mean()
            val_loss += loss.item()

    val_loss_tensor = torch.tensor(val_loss).to(device)
    dist.reduce(val_loss_tensor, op=dist.ReduceOp.SUM, dst=0)
    return val_loss_tensor.item()

def save_checkpoint(model_engine, tokenizer, checkpoint_dir, epoch, global_step, rank):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch + 1}-step{global_step}')
    if rank == 0:
        os.makedirs(checkpoint_path, exist_ok=True)
    dist.barrier()
    model_engine.save_checkpoint(checkpoint_path)
    if rank == 0:
        tokenizer.save_pretrained(checkpoint_path)
    dist.barrier()

def log_validation_metrics(writer, avg_train_loss, avg_val_loss, perplexity, epoch):
    writer.add_scalar('Validation/Loss', avg_val_loss, epoch + 1)
    writer.add_scalar('Validation/Perplexity', perplexity, epoch + 1)

def log_epoch_summary(epoch, avg_train_loss, avg_val_loss, perplexity, epoch_start_time):
    epoch_duration = datetime.now() - epoch_start_time
    metrics = {
        "epoch": epoch + 1,
        "training_loss": avg_train_loss,
        "validation_loss": avg_val_loss,
        "perplexity": perplexity,
        "epoch_duration": str(epoch_duration)
    }
    log_epoch_metrics(epoch + 1, metrics)
    logging.info(f"Epoch {epoch + 1} completed. Avg Training Loss: {avg_train_loss:.4f}, "
                 f"Avg Validation Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")

def handle_best_model(avg_val_loss, best_val_loss, model_engine, tokenizer, config, rank):
    is_best = avg_val_loss < best_val_loss
    is_best_tensor = torch.tensor(int(is_best), device=model_engine.device)
    dist.broadcast(is_best_tensor, src=0)
    is_best = bool(is_best_tensor.item())

    if is_best:
        trained_models_dir = get_config_value(config, ['paths', 'trained_models_dir'], required=True)
        model_name = get_config_value(config, ['model', 'name'], required=True)
        model_save_dir = os.path.join(
            trained_models_dir,
            f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if rank == 0:
            os.makedirs(model_save_dir, exist_ok=True)
        dist.barrier()
        model_engine.save_checkpoint(model_save_dir)
        if rank == 0:
            tokenizer.save_pretrained(model_save_dir)
        dist.barrier()

    return is_best