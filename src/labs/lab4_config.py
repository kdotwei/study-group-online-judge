from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data-lab4"

SEQ_LEN = 1024
TOKENIZER_DIR = DATA_ROOT / "tokenizer"
VALIDATION_DIR = DATA_ROOT / "c4-validation"

DATA_PROFILES = {
    "spark": {
        "target_tokens": 240_000_000,
        "path": DATA_ROOT / "c4-tokenized-spark",
    },
    "prologue": {
        "target_tokens": 380_000_000,
        "path": DATA_ROOT / "c4-tokenized-prologue"
    }
}

def get_data_profile(experiment):
    series = experiment.split("-", 1)[0]
    
    if series not in DATA_PROFILES:
        raise ValueError(f"Unknown experient series: {series}")
    
    return DATA_PROFILES[series]