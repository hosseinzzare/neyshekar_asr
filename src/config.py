"""
Centralized Configuration & Reproducibility Management for Whisper Large-v3 Fine-tuning.
Task 2 - Step 2: Configuration & Deterministic Seed Management.
"""

import os
import random
import numpy as np


class Config:
    # -------------------------------------------------------------
    # 1. Reproducibility & Seed
    # -------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------
    # 2. File & Directory Paths
    # -------------------------------------------------------------
    TRAIN_CSV = "data/train.csv"
    VAL_CSV = "data/val.csv"
    OUTPUT_DIR = "./whisper-large-v3-neyshekar-qlora"
    LOGS_DIR = "./logs"

    # -------------------------------------------------------------
    # 2b. HuggingFace Hub Dataset Source (for audio loading)
    # -------------------------------------------------------------
    HF_DATASET_NAME = "shekar-ai/neyshekar-v4-persian-asr-fa"
    MAP_WRITER_BATCH_SIZE = 500  # Flush Arrow cache every N samples to prevent OOM during .map()

    # -------------------------------------------------------------
    # 3. Model & Language Settings
    # -------------------------------------------------------------
    MODEL_NAME_OR_PATH = "openai/whisper-large-v3"
    LANGUAGE = "persian"
    TASK = "transcribe"
    SAMPLING_RATE = 16000

    # -------------------------------------------------------------
    # 4. QLoRA Optimization (4-bit Quantization + PEFT)
    # -------------------------------------------------------------
    LOAD_IN_4BIT = True
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_COMPUTE_DTYPE = "float16"
    BNB_4BIT_USE_DOUBLE_QUANT = True
    
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj"]

    # -------------------------------------------------------------
    # 5. Training Arguments
    # -------------------------------------------------------------
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-3
    WARMUP_STEPS = 50
    PER_DEVICE_TRAIN_BATCH_SIZE = 8
    PER_DEVICE_EVAL_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective Batch Size = 16
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 100
    SAVE_STEPS = 100
    SAVE_TOTAL_LIMIT = 2
    METRIC_FOR_BEST_MODEL = "wer"
    GREATER_IS_BETTER = False
    LOAD_BEST_MODEL_AT_END = True
    MAX_STEPS = -1  # Set to 30 for quick smoke test on Google Colab


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets random seed across Python, NumPy, PyTorch, and CUDA for complete reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[SEED] Seed set to {seed} across Python, NumPy, PyTorch, and CUDA.")
    except ImportError:
        print(f"[SEED] Seed set to {seed} across Python and NumPy (PyTorch not loaded).")


# Module-level upper-case variables for direct import compatibility
SEED = Config.SEED
TRAIN_CSV = Config.TRAIN_CSV
VAL_CSV = Config.VAL_CSV
OUTPUT_DIR = Config.OUTPUT_DIR
LOGS_DIR = Config.LOGS_DIR
HF_DATASET_NAME = Config.HF_DATASET_NAME
MAP_WRITER_BATCH_SIZE = Config.MAP_WRITER_BATCH_SIZE

MODEL_NAME_OR_PATH = Config.MODEL_NAME_OR_PATH
LANGUAGE = Config.LANGUAGE
TASK = Config.TASK
SAMPLING_RATE = Config.SAMPLING_RATE

LOAD_IN_4BIT = Config.LOAD_IN_4BIT
BNB_4BIT_QUANT_TYPE = Config.BNB_4BIT_QUANT_TYPE
BNB_4BIT_COMPUTE_DTYPE = Config.BNB_4BIT_COMPUTE_DTYPE
BNB_4BIT_USE_DOUBLE_QUANT = Config.BNB_4BIT_USE_DOUBLE_QUANT

LORA_R = Config.LORA_R
LORA_ALPHA = Config.LORA_ALPHA
LORA_DROPOUT = Config.LORA_DROPOUT
TARGET_MODULES = Config.TARGET_MODULES

NUM_EPOCHS = Config.NUM_EPOCHS
LEARNING_RATE = Config.LEARNING_RATE
WARMUP_STEPS = Config.WARMUP_STEPS
PER_DEVICE_TRAIN_BATCH_SIZE = Config.PER_DEVICE_TRAIN_BATCH_SIZE
PER_DEVICE_EVAL_BATCH_SIZE = Config.PER_DEVICE_EVAL_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = Config.GRADIENT_ACCUMULATION_STEPS
FP16 = Config.FP16
GRADIENT_CHECKPOINTING = Config.GRADIENT_CHECKPOINTING
EVAL_STRATEGY = Config.EVAL_STRATEGY
EVAL_STEPS = Config.EVAL_STEPS
SAVE_STEPS = Config.SAVE_STEPS
SAVE_TOTAL_LIMIT = Config.SAVE_TOTAL_LIMIT
METRIC_FOR_BEST_MODEL = Config.METRIC_FOR_BEST_MODEL
GREATER_IS_BETTER = Config.GREATER_IS_BETTER
LOAD_BEST_MODEL_AT_END = Config.LOAD_BEST_MODEL_AT_END
MAX_STEPS = Config.MAX_STEPS

if __name__ == '__main__':
    set_seed(SEED)
    print("Configuration loaded successfully:")
    print(f"  - Model: {MODEL_NAME_OR_PATH}")
    print(f"  - Language: {LANGUAGE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Train Batch Size: {PER_DEVICE_TRAIN_BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} (Effective: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  - QLoRA Rank: {LORA_R}, Alpha: {LORA_ALPHA}")
