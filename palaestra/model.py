import os
import logging
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from palaestra.config import get_config_value

def setup_model_and_tokenizer(config, args):
    model_name = get_config_value(config, ['model', 'name'], required=True)
    models_to_train_dir = get_config_value(config, ['paths', 'models_to_train_dir'], required=True)
    model_path = os.path.join(models_to_train_dir, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    special_tokens_dict = get_config_value(config, ['model', 'special_tokens'], default={})
    if special_tokens_dict:
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.gradient_checkpointing_enable()

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": get_config_value(config, ['training', 'weight_decay'], default=0.0),
        },
    ]

    deepspeed_config_path = get_config_value(config, ['paths', 'deepspeed_config'], default='config/ds_config.json')
    with open(deepspeed_config_path, 'r') as f:
        ds_config = json.load(f)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters,
        config_params=ds_config
    )

    return model, tokenizer, model_engine, optimizer, lr_scheduler