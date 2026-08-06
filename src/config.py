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
    # 3b. Audio Preprocessing (the Task 1 investigation called for these)
    # -------------------------------------------------------------
    # Task 1 found 27.9% of clips with a suspiciously low character-per-second rate (long
    # silences) and 22.1% touching the digital ceiling (clipping). Whisper's autoregressive
    # decoder is known to hallucinate text over silence, so leading/trailing silence is trimmed
    # before feature extraction. Only the EDGES are trimmed -- pauses inside an utterance are
    # natural speech and are deliberately preserved.
    ENABLE_VAD_TRIM = True
    VAD_TOP_DB = 40.0          # frames quieter than peak-40dB count as silence
    VAD_MARGIN_MS = 100.0      # keep 100 ms either side so word onsets are never clipped
    VAD_MIN_DURATION_S = 1.0   # if trimming would leave less than this, keep the original

    # Peak-normalise every clip to a common level. Cannot repair already-clipped audio (that
    # information is gone), but it gives the dataset a consistent dynamic range across speakers.
    ENABLE_PEAK_NORM = True
    PEAK_NORM_DB = -3.0

    # -------------------------------------------------------------
    # 4. QLoRA Optimization (4-bit Quantization + PEFT)
    # -------------------------------------------------------------
    # 4-bit quantization exists to fit large models on SMALL GPUs. It is not free: bitsandbytes
    # must dequantize every weight back to 16-bit on each forward pass, which costs compute.
    # On a 24 GB L4 the full bf16 model (~3.1 GB) fits comfortably, so quantization buys ~2.2 GB
    # of memory we do not need in exchange for per-step overhead -- and LoRA then trains on
    # APPROXIMATED weights rather than exact ones. Use --no_quantization to disable it.
    USE_QUANTIZATION = True
    LOAD_IN_4BIT = True
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_COMPUTE_DTYPE = "float16"
    BNB_4BIT_USE_DOUBLE_QUANT = True

    # Autocast dtype used when quantization is OFF. The weights stay fp32 (so LoRA and the
    # optimizer are numerically stable) while matmuls run at 16-bit on tensor cores. bf16 has
    # the same exponent range as fp32, so it needs no loss scaling; requires SM 8.0+ (Ampere
    # and newer -- the L4 is SM 8.9).
    USE_BF16_WHEN_UNQUANTIZED = True
    
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
    # Gradient checkpointing recomputes activations during the backward pass instead of storing
    # them: much lower VRAM, but ~1.3-1.5x slower. Required on a 16 GB T4; on a 24 GB L4 it can
    # be disabled with --no_gradient_checkpointing for a free speedup (gradients are identical).
    GRADIENT_CHECKPOINTING = True
    DATALOADER_NUM_WORKERS = 2
    EVAL_STRATEGY = "steps"
    # With ~2,089 steps/epoch (33,432 train rows / effective batch 16), 3 epochs is ~6,267 steps.
    # EVAL_STEPS=100 would mean 62 evaluations; each one runs autoregressive generate() over the
    # full 5,900-row validation set, which costs roughly 4-8 HOURS of pure evaluation overhead on
    # top of training. 500 gives ~12 evaluation points -- still plenty of resolution for the
    # Task 3 loss/WER/CER curves, at a fraction of the cost.
    EVAL_STEPS = 500
    SAVE_STEPS = 500

    # Cap how many validation rows are used for the PERIODIC in-training evaluations.
    # A fixed 1,500-row subset (deterministically sampled with SEED) keeps every eval point
    # comparable while cutting generate() cost ~4x. Set to None to always use the full split.
    # NOTE: the final reported metrics should be computed on the FULL validation set -- see
    # the --final_full_eval flag in train.py, which does exactly that once at the end.
    MAX_EVAL_SAMPLES = 1500
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

USE_QUANTIZATION = Config.USE_QUANTIZATION
USE_BF16_WHEN_UNQUANTIZED = Config.USE_BF16_WHEN_UNQUANTIZED
ENABLE_VAD_TRIM = Config.ENABLE_VAD_TRIM
VAD_TOP_DB = Config.VAD_TOP_DB
VAD_MARGIN_MS = Config.VAD_MARGIN_MS
VAD_MIN_DURATION_S = Config.VAD_MIN_DURATION_S
ENABLE_PEAK_NORM = Config.ENABLE_PEAK_NORM
PEAK_NORM_DB = Config.PEAK_NORM_DB
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
MAX_EVAL_SAMPLES = Config.MAX_EVAL_SAMPLES
DATALOADER_NUM_WORKERS = Config.DATALOADER_NUM_WORKERS
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
