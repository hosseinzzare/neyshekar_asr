"""
Root config launcher script.
Re-exports configuration variables from src/config.py
"""
import sys
import os

from src.config import (
    Config,
    set_seed,
    SEED,
    TRAIN_CSV,
    VAL_CSV,
    OUTPUT_DIR,
    LOGS_DIR,
    HF_DATASET_NAME,
    MAP_WRITER_BATCH_SIZE,
    MODEL_NAME_OR_PATH,
    LANGUAGE,
    TASK,
    SAMPLING_RATE,
    LOAD_IN_4BIT,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    NUM_EPOCHS,
    LEARNING_RATE,
    WARMUP_STEPS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    FP16,
    GRADIENT_CHECKPOINTING,
    EVAL_STRATEGY,
    EVAL_STEPS,
    SAVE_STEPS,
    MAX_EVAL_SAMPLES,
    DATALOADER_NUM_WORKERS,
    SAVE_TOTAL_LIMIT,
    METRIC_FOR_BEST_MODEL,
    GREATER_IS_BETTER,
    LOAD_BEST_MODEL_AT_END,
    MAX_STEPS
)

if __name__ == '__main__':
    set_seed(SEED)
    print("Configuration loaded successfully via root config.py:")
    print(f"  - Model: {MODEL_NAME_OR_PATH}")
    print(f"  - Language: {LANGUAGE}")
    print(f"  - Output Dir: {OUTPUT_DIR}")
    print(f"  - QLoRA Rank (R): {LORA_R}, Alpha: {LORA_ALPHA}")
