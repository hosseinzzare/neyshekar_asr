# 🎙️ Neyshekar ASR - Whisper Large-v3 Fine-Tuning Pipeline (QLoRA 4-bit)

A production-ready, highly optimized end-to-end pipeline for Persian speech data preprocessing and fine-tuning **`openai/whisper-large-v3`** using **QLoRA (Quantized Low-Rank Adaptation)** and **4-Bit NF4 Quantization**.

---

## 📌 Key Features

1. **Speech Preprocessing & Data Cleaning (Task 1):**
   - **Deduplication:** Exact and approximate deduplication across audio and transcript samples.
   - **Audio Validation:** Verifies audio integrity, sample rates (16 kHz mono), duration bounds, and corrupt file detection.
   - **Speech Rate Filtering:** Filters out samples outside reasonable speech rates (1.5 to 22 characters per second).
   - **Persian Text Normalization:** Arabic-to-Persian character conversion, digits-to-words transformation (`num2fawords`), punctuation stripping, and whitespace normalization.
   - **Balanced Data Splitting:** Stratified split into **85% Training** (`data/train.csv`) and **15% Validation** (`data/val.csv`).

2. **QLoRA Fine-Tuning Architecture (Task 2):**
   - **4-Bit NF4 Quantization:** Loads base model `openai/whisper-large-v3` with `bitsandbytes` 4-bit quantization for minimal VRAM footprint (~10–12 GB VRAM required).
   - **PEFT LoRA Adapters:** Injects low-rank adapters (`r=32`, `alpha=64`) targeting key attention projection layers (`q_proj`, `v_proj`).
   - **Gradient Checkpointing Hook:** Registers `make_inputs_require_grad` forward hook on encoder `conv1` layer to enable gradient flow during QLoRA checkpointing.
   - **PEFT+Whisper Keyword Collision Fix:** Includes custom `safe_base_forward` wrapper preventing `input_ids` parameter collision in `WhisperDecoder`.
   - **Automated Evaluation Metrics:** Computes normalized **WER (Word Error Rate)** and **CER (Character Error Rate)** during training using Hugging Face `evaluate` and `jiwer`.

---

## 📂 Project Structure

```text
neyshekar_asr/
├── config.py             # Global configurations (hyperparameters, file paths, random seed)
├── data_prep.py          # Entry point for Task 1 (Data preprocessing and cleaning)
├── dataset.py            # Dataset loader & Whisper DataCollator with dynamic padding
├── model.py              # Whisper Large-v3 4-bit QLoRA model initialization
├── metrics.py            # Evaluation metrics (WER & CER computation)
├── train.py              # Entry point for Task 2 (Training pipeline & CLI args)
├── requirements.txt      # PyPI Python dependencies
├── README.md             # Project documentation
├── data/                 # Directory containing preprocessed CSV files
│   ├── train.csv         # Training dataset (85%)
│   └── val.csv           # Validation dataset (15%)
└── src/                  # Internal pipeline modules
    ├── config.py
    ├── data_prep.py
    ├── dataset.py
    ├── metrics.py
    ├── model.py
    ├── normalizer.py
    ├── text_cleaner.py
    └── train.py
```

---

## ⚡ Setup & Installation

### 1. Clone Repository & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/hosseinzzare/neyshekar_asr.git
cd neyshekar_asr

# Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux / macOS
# or
.venv\Scripts\activate     # On Windows

# Install required dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### Step 1: Preprocess Data (Task 1)

If you wish to re-generate the cleaned datasets from raw CSV files:

```bash
python data_prep.py
```
> **Output:** Generates `data/train.csv` (85%) and `data/val.csv` (15%).

---

### Step 2: Quick Smoke Test (30 Steps Verification)

To verify model initialization, CUDA memory management, and metric calculation prior to full training:

```bash
python train.py --max_steps 30 --output_dir ./whisper-large-v3-smoke-test
```

---

### Step 3: Full Fine-Tuning (Full Training Run)

To run complete fine-tuning on all training epochs:

```bash
python train.py --max_steps -1 --epochs 3 --output_dir ./whisper-large-v3-neyshekar-qlora
```

---

## ⚙️ CLI Arguments (`train.py`)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--max_steps` | `-1` | Maximum training steps (`-1` for full training across all epochs) |
| `--epochs` | `3` | Total number of training epochs |
| `--output_dir` | `./whisper-large-v3-neyshekar-qlora` | Directory to save checkpoints and final model weights |
| `--train_csv` | `data/train.csv` | Path to training CSV dataset |
| `--val_csv` | `data/val.csv` | Path to validation CSV dataset |

---

## 📊 Training Metrics & Logging

Training metrics, loss curves, WER, and CER evaluation logs are saved under `./logs`.  
To launch **TensorBoard**:

```bash
tensorboard --logdir ./logs
```

---

## 🛡️ VRAM & Memory Efficiency

- **Hardware Requirement:** Minimum **12 GB GPU VRAM** (e.g., RTX 3060, RTX 4070, T4, V100, A100).
- **Default Batch Config:** `PER_DEVICE_TRAIN_BATCH_SIZE = 8` with `GRADIENT_ACCUMULATION_STEPS = 4` (effective batch size = 32).
- **Gradient Checkpointing & Mixed Precision:** `FP16 = True` with gradient checkpointing enabled for peak memory optimization.

---
Developed for Persian Speech Recognition Assessment (Neyshekar ASR).
