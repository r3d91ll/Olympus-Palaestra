from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str, model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer