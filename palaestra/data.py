import json
import random
import logging
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from palaestra.config import get_config_value

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        input_text = record.get('input_text', '')
        target_text = record.get('target_text', '')
        full_text = f"{input_text}\n{target_text}"
        return full_text

def load_and_split_data(config):
    dataset_path = get_config_value(config, ['paths', 'preprocessed_dataset'], required=True)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    random.shuffle(data)
    
    validation_split = get_config_value(config, ['training', 'validation_split'], default=0.1)
    split_index = int(len(data) * (1 - validation_split))
    return data[:split_index], data[split_index:]

def get_collate_fn(tokenizer, max_length):
    def collate_fn(batch_texts):
        tokenized_batch = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding='longest',
            return_tensors='pt'
        )
        labels = tokenized_batch['input_ids'].clone()
        tokenized_batch['labels'] = labels
        return tokenized_batch
    return collate_fn

def prepare_datasets(config, tokenizer):
    train_data, val_data = load_and_split_data(config)
    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)

    batch_size = get_config_value(config, ['training', 'batch_size'], default=4)
    validation_batch_size = get_config_value(config, ['training', 'validation_batch_size'], default=1)
    num_workers = get_config_value(config, ['training', 'num_workers'], default=4)
    max_length = get_config_value(config, ['training', 'max_sequence_length'], default=1024)

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

    return train_loader, val_loader