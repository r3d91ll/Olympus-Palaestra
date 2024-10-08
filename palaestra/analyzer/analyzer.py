import json
import os
import logging
from collections import Counter
from transformers import AutoConfig, AutoTokenizer, AutoModel
import numpy as np
from .utils import save_report, save_sequence_length_histogram

def generate_report(model_path, dataset_path, output_dir, num_samples):
    report = {
        "model_info": {},
        "dataset_info": {},
        "compatibility_checks": [],
        "warnings_and_errors": []
    }

    # Load model configuration and tokenizer
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}. Please check if the path is correct and accessible.")
        
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load the model architecture without weights to save memory
        model = AutoModel.from_config(config)
        
        report["model_info"] = {
            "model_name": config.name_or_path,
            "model_type": config.model_type,
            "num_layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "vocab_size": config.vocab_size,
        }
    except Exception as e:
        error_message = f"Failed to load model configuration or tokenizer: {e}"
        report["warnings_and_errors"].append({"error": error_message})
        logging.error(error_message, exc_info=True)
        save_report(report, output_dir)
        return

    # Estimate total parameters using the model's parameters
    try:
        total_parameters = sum(p.numel() for p in model.parameters())
        report["model_info"]["total_parameters"] = total_parameters
        
        parameter_dtype_size = next(model.parameters()).element_size()
        estimated_memory_gb = (total_parameters * parameter_dtype_size) / (1024 ** 3)
        report["model_info"]["estimated_memory_gb"] = round(estimated_memory_gb, 2)
    except Exception as e:
        error_message = f"Failed to estimate model parameters: {e}"
        report["warnings_and_errors"].append({"error": error_message})
        logging.error(error_message, exc_info=True)

    # Analyze training dataset
    if not os.path.exists(dataset_path):
        error_message = f"Dataset file not found: {dataset_path}. Please check if the file path is correct and accessible."
        report["warnings_and_errors"].append({"error": error_message})
        logging.error(error_message)
        save_report(report, output_dir)
        return

    try:
        samples = []
        sequence_lengths = []
        token_counter = Counter()
        field_lengths = {}
        oov_token_count = 0
        total_token_count = 0
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= num_samples:
                    break
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                    for key, value in sample.items():
                        if isinstance(value, str):
                            # Tokenize the text
                            encoding = tokenizer(
                                value,
                                add_special_tokens=True,
                                truncation=False,
                                return_attention_mask=False,
                                return_token_type_ids=False
                            )
                            tokens = encoding["input_ids"]
                            length = len(tokens)
                            sequence_lengths.append(length)
                            token_counter.update(tokens)
                            
                            # Update field_lengths
                            if key not in field_lengths:
                                field_lengths[key] = []
                            field_lengths[key].append(length)
                            
                            # Count OOV tokens
                            oov_token_id = tokenizer.unk_token_id
                            oov_token_count += tokens.count(oov_token_id)
                            total_token_count += len(tokens)
                except json.JSONDecodeError as e:
                    error_message = f"JSON decoding error in sample {idx}: {e}"
                    report["warnings_and_errors"].append({"error": error_message})
                    logging.error(error_message)
                    continue
        
        if sequence_lengths:
            average_sequence_length = np.mean(sequence_lengths)
            max_sequence_length = max(sequence_lengths)
            percentiles = np.percentile(sequence_lengths, [50, 75, 90, 95, 99]).tolist()
            
            report["dataset_info"].update({
                "num_samples_analyzed": len(samples),
                "average_sequence_length": average_sequence_length,
                "max_sequence_length": max_sequence_length,
                "sequence_length_percentiles": {
                    "50th": percentiles[0],
                    "75th": percentiles[1],
                    "90th": percentiles[2],
                    "95th": percentiles[3],
                    "99th": percentiles[4]
                },
                "vocabulary_size": len(token_counter),
                "sample_keys": list(samples[0].keys()) if samples else []
            })
            
            # Field-specific statistics
            for field, lengths in field_lengths.items():
                average_length = np.mean(lengths)
                max_length = max(lengths)
                field_percentiles = np.percentile(lengths, [50, 75, 90, 95, 99]).tolist()
                report["dataset_info"][f"{field}_length_stats"] = {
                    "average_length": average_length,
                    "max_length": max_length,
                    "sequence_length_percentiles": {
                        "50th": field_percentiles[0],
                        "75th": field_percentiles[1],
                        "90th": field_percentiles[2],
                        "95th": field_percentiles[3],
                        "99th": field_percentiles[4]
                    }
                }
            
            # OOV token analysis
            oov_token_percentage = (oov_token_count / total_token_count) * 100 if total_token_count > 0 else 0
            report["dataset_info"]["oov_token_percentage"] = oov_token_percentage
            
            # Compatibility checks
            if hasattr(config, "max_position_embeddings") and config.max_position_embeddings:
                num_exceeding = sum(1 for l in sequence_lengths if l > config.max_position_embeddings)
                percentage_exceeding = (num_exceeding / len(sequence_lengths)) * 100
                
                if num_exceeding > 0:
                    report["compatibility_checks"].append({
                        "warning": f"{num_exceeding} out of {len(sequence_lengths)} samples ({percentage_exceeding:.2f}%) exceed the model's max position embeddings ({config.max_position_embeddings})."
                    })
                    report["compatibility_checks"].append({
                        "recommendation": (
                            "Consider truncating or splitting sequences longer than the max position embeddings. "
                            "Alternatively, you may adjust the model's max position embeddings if possible."
                        )
                    })
                else:
                    report["compatibility_checks"].append({
                        "info": "All sequences are within the model's max position embeddings."
                    })
            else:
                report["compatibility_checks"].append({
                    "info": "Model does not define max position embeddings."
                })
            
            # Check for required special tokens
            required_special_tokens = [tokenizer.cls_token, tokenizer.sep_token]
            missing_special_tokens = [tok for tok in required_special_tokens if tok is None]
            
            if missing_special_tokens:
                report["compatibility_checks"].append({
                    "warning": f"The tokenizer is missing special tokens: {missing_special_tokens}."
                })
                report["compatibility_checks"].append({
                    "recommendation": "Update the tokenizer to include the required special tokens."
                })
            
        else:
            report["warnings_and_errors"].append({
                "error": "No valid sequences found in the dataset."
            })
        
    except Exception as e:
        error_message = f"Failed to analyze dataset: {e}"
        report["warnings_and_errors"].append({"error": error_message})
        logging.error(error_message, exc_info=True)
    
    save_report(report, output_dir)
    
    # Optionally, save a histogram of sequence lengths
    save_sequence_length_histogram(sequence_lengths, output_dir)