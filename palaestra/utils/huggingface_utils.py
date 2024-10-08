import os
import logging
from huggingface_hub import HfApi, Repository
from transformers import AutoTokenizer, AutoModel
from ..config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_model_and_dataset():
    logging.info("Ensuring model and dataset availability")
    model_name = config.get('model', 'name')
    dataset_name = config.get('huggingface', 'dataset_repo')
    models_dir = config.get('paths', 'models_to_train_dir')
    datasets_dir = config.get('paths', 'datasets_dir')

    ensure_model(model_name, models_dir)
    ensure_dataset(dataset_name, datasets_dir)
    logging.info("Model and dataset availability confirmed")

def ensure_model(model_name, models_dir):
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        logging.info(f"Model {model_name} not found locally. Downloading...")
        download_model(model_name, model_path)
    else:
        logging.info(f"Model {model_name} found locally. Checking for updates...")
        update_model(model_name, model_path)

def ensure_dataset(dataset_name, datasets_dir):
    dataset_path = os.path.join(datasets_dir, dataset_name)
    if not os.path.exists(dataset_path):
        logging.info(f"Dataset {dataset_name} not found locally. Downloading...")
        download_dataset(dataset_name, dataset_path)
    else:
        logging.info(f"Dataset {dataset_name} found locally. Checking for updates...")
        update_dataset(dataset_name, dataset_path)

def download_model(model_name, model_path):
    try:
        AutoModel.from_pretrained(model_name, cache_dir=model_path)
        AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
        logging.info(f"Model {model_name} downloaded successfully")
    except Exception as e:
        logging.error(f"Error downloading model {model_name}: {e}")
        raise

def update_model(model_name, model_path):
    try:
        api = HfApi()
        repo_info = api.repo_info(repo_id=model_name, repo_type="model")
        local_commit_hash = get_local_commit_hash(model_path)
        if local_commit_hash != repo_info.sha:
            logging.info(f"Updating model {model_name}")
            download_model(model_name, model_path)
        else:
            logging.info(f"Model {model_name} is up to date")
    except Exception as e:
        logging.error(f"Error updating model {model_name}: {e}")
        raise

def download_dataset(dataset_name, dataset_path):
    try:
        repo = Repository(local_dir=dataset_path, clone_from=dataset_name)
        repo.git_pull()
        logging.info(f"Dataset {dataset_name} downloaded successfully")
    except Exception as e:
        logging.error(f"Error downloading dataset {dataset_name}: {e}")
        raise

def update_dataset(dataset_name, dataset_path):
    try:
        repo = Repository(local_dir=dataset_path, clone_from=dataset_name)
        if repo.is_repo_clean():
            logging.info(f"Dataset {dataset_name} is up to date")
        else:
            repo.git_pull()
            logging.info(f"Dataset {dataset_name} updated successfully")
    except Exception as e:
        logging.error(f"Error updating dataset {dataset_name}: {e}")
        raise

def get_local_commit_hash(model_path):
    commit_hash_file = os.path.join(model_path, "refs", "main")
    if os.path.exists(commit_hash_file):
        with open(commit_hash_file, 'r') as f:
            return f.read().strip()
    return None

def load_model_and_tokenizer(model_name, local_dir):
    logging.info(f"Loading model and tokenizer from {local_dir}")
    try:
        model = AutoModel.from_pretrained(local_dir)
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        logging.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise