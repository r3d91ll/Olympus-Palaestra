import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any, Tuple

class TextDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: PreTrainedTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}

def create_dataloaders(dataset: TextDataset, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    # Calculate the split sizes
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Create the splits
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader