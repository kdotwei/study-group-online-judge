import os
import time
import math
import torch
import argparse
from datasets import load_from_disk
from transformers import GPT2Config, GPT2LMHeadModel, set_seed
from transformers import TrainingArguments
from transformers import Trainer
from transformers import TrainerCallback
from transformers import AutoTokenizer

class TimeLimitCallback(TrainerCallback):
    def __init__(self):
        start = float(os.environ.get("LAB4_JOB_START_TIME", time.time()))
        self.deadline = start + 25*60
    
    def on_step_end(self, args, state, control, **kwargs):
        stop = torch.tensor(
            int(time.time() >= self.deadline),
            device=args.device,
        )
        
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
            
        if stop.item():
            control.should_training_stop = True
            control.should_save = True
            control.should_log = True
            control.should_evaluate = False
        
        return control

def prepare_batch(samples):
    input_ids = torch.tensor(
        [sample["input_ids"] for sample in samples],
        dtype=torch.long,
    )
    batch = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
    }
    return batch

# Load training data
print("Training data loading...")
train_dataset = load_from_disk("data-lab4/c4-tokenized")

# Initialize shuffling weights GPT2 small
print("Initializing GPT2 small...")
set_seed(42)
config = GPT2Config(use_cache=False)
model = GPT2LMHeadModel(config)

# Set up training arguments
print("Setting up training arguments...")
os.environ["WANDB_PROJECT"] = "gpt2-training"
experiment = os.environ.get["EXPERIMENT", "spark-e1"]
job_id = os.environ.get["SLURM_JOB_ID", "local"]
run_name = f"{experiment}-{job_id}"

training_args = TrainingArguments(
    output_dir=f"trainings/lab4/{run_name}",
    run_name=run_name,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,
    learning_rate=2.5e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    bf16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to=["wandb"],
    ddp_find_unused_parameters=False,
)

print("Training arguments ready")

# Validate data
print("Loading validation data...")
eval_dataset = load_from_disk("data-lab4/c4-validation")
training_args.eval_strategy = "steps"
training_args.eval_steps = 100
training_args.per_device_eval_batch_size = 16
training_args.prediction_loss_only = True

# Build trainer
print("Building trainer...")
tokenizer = AutoTokenizer.from_pretrained("data-lab4/tokenizer", local_files_only=True)
training_args.include_num_input_tokens_seen = "all"

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=prepare_batch,
    processing_class=tokenizer,
)

print("Validation blocks: ", len(eval_dataset))
print("Trainer ready")

# Training time control
print("Adding time limit controller...")
trainer.add_callback(TimeLimitCallback())
print("Time limit callback ready")

# Add entry point and final save
print("Adding entry point and final save...")
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

if args.train:
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    metrics = trainer.evaluate()
    loss = metrics["eval_loss"]
    metrics["eval_perplexity"] = math.exp(loss) if loss < 700 else math.inf
    trainer.log(metrics)
    trainer.save_metrics("eval", metrics)
