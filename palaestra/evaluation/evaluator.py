import torch
from torch.utils.data import DataLoader

def evaluate_model(model, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)