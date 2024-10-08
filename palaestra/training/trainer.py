import deepspeed
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from typing import Dict, Any

class DeepSpeedTrainer:
    def __init__(self, model: PreTrainedModel, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any], args: Any):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.args = args

        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            config=config['deepspeed_config']
        )

    def train_epoch(self, epoch: int):
        self.model_engine.train()
        for batch in self.train_loader:
            loss = self.model_engine(**batch).loss
            self.model_engine.backward(loss)
            self.model_engine.step()

    def validate(self, epoch: int):
        self.model_engine.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.model_engine(**batch).loss
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, path: str):
        self.model_engine.save_checkpoint(path)