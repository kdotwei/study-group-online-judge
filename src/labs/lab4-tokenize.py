import os
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from lab4_config import SEQ_LEN, TOKENIZER_DIR, get_data_profile

experiment = os.environ.get("EXPERIMENT", "spark")
profile = get_data_profile(experiment)
TARGET_TOKENS = profile["target_tokens"]
output_path = profile["path"]

if output_path.exists():
    raise FileExistsError(f"Dataset already exists: {output_path}")

tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_file_only=True)

def generate_blocks():
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    buffer = []
    saved_tokens = 0

    for sample in dataset:
        ids = tokenizer(sample["text"], add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        buffer.append(tokenizer.eos_token_id)

        while len(buffer) >= SEQ_LEN:
            block = buffer[:SEQ_LEN]
            buffer = buffer[SEQ_LEN:]
            yield {"input_ids": block} # Packing
            saved_tokens += SEQ_LEN

            if saved_tokens >= TARGET_TOKENS:
                return

tokenized = Dataset.from_generator(generate_blocks)

print("blocks:", len(tokenized))
print("tokens:", len(tokenized) * SEQ_LEN)

tokenized.save_to_disk(str(output_path))