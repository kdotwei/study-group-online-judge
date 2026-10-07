from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parents[2]
model_path = root / "trainings/lab4-gpt2"
repo_id = "cerulean-works/kdotwei-lab4-gpt2-spark"

model = AutoModelForCausalLM.from_pretrained(
    model_path, local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True
)

model.config.use_cache = True
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print(f"Uploaded: https://huggingface.co/{repo_id}")
