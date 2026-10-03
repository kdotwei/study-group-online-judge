from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

SEQ_LEN = 1024
TARGET_TOKENS = 240_000_000

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

tokenized.save_to_disk("data-lab4/c4-tokenized")