import deepspeed
import torch
from typing import Dict, Any

def initialize_deepspeed(model, optimizer_params, args, ds_config):
    """
    Initialize DeepSpeed engine.
    """
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_params,
        config=ds_config
    )
    return model_engine, optimizer

def get_optimizer_params(model, weight_decay):
    """
    Prepare optimizer parameters for DeepSpeed.
    """
    return [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": weight_decay,
        },
    ]

def load_ds_config(config_path: str) -> Dict[str, Any]:
    """
    Load DeepSpeed configuration from a JSON file.
    """
    import json
    with open(config_path, 'r') as f:
        return json.load(f)

def save_deepspeed_checkpoint(model_engine, path: str):
    """
    Save a DeepSpeed checkpoint.
    """
    model_engine.save_checkpoint(path)

def load_deepspeed_checkpoint(model_engine, path: str):
    """
    Load a DeepSpeed checkpoint.
    """
    _, client_state = model_engine.load_checkpoint(path)
    return client_state