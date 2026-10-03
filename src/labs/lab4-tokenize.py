from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
dataset = load_dataset("allenai/c4", "en", split="train")

def tokenize_fn(batch):
    return tokenizer(batch["text"], add_special_tokens=False)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names, num_proc=8)
tokenized.save_to_disk("data-lab4/c4-tokenized")
