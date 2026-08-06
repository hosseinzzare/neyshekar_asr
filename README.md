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
   - **Silence Trimming (VAD):** Energy-based removal of leading/trailing silence before feature extraction, addressing the 27.9% of clips with a low character-per-second rate. Whisper's decoder is known to hallucinate text over silence. Pauses *inside* an utterance are deliberately preserved as natural speech rhythm.
   - **Peak Normalisation:** Every clip scaled to −3 dBFS, giving a consistent dynamic range across the 22.1% of clips that touch the digital ceiling. This is a pure scalar multiply, so no distortion is introduced and the output can never clip.
   - *Note:* neither step reduces training time — Whisper pads every clip to 30 s regardless. They are quality measures, not speed measures.

2. **QLoRA Fine-Tuning Architecture (Task 2):**
   - **4-Bit NF4 Quantization:** Loads base model `openai/whisper-large-v3` with `bitsandbytes` 4-bit quantization for minimal VRAM footprint (~10–12 GB VRAM required).
   - **PEFT LoRA Adapters:** Injects low-rank adapters (`r=32`, `alpha=64`) targeting key attention projection layers (`q_proj`, `v_proj`).
   - **Gradient Checkpointing Hook:** Registers `make_inputs_require_grad` forward hook on encoder `conv1` layer to enable gradient flow during QLoRA checkpointing.
   - **PEFT+Whisper Keyword Collision Fix:** Includes custom `safe_base_forward` wrapper preventing `input_ids` parameter collision in `WhisperDecoder`.
   - **Automated Evaluation Metrics:** Computes normalized **WER (Word Error Rate)** and **CER (Character Error Rate)** during training using Hugging Face `evaluate` and `jiwer`.

3. **Data Loading Architecture:**
   - **Audio:** Loaded from HuggingFace Hub (`shekar-ai/neyshekar-v4-persian-asr-fa`) which contains actual audio waveforms.
   - **Text Labels:** Loaded from local CSV files (`data/train.csv`, `data/val.csv`) containing cleaned Persian transcripts from Task 1 pipeline.
   - **Merge Strategy:** Samples are matched by the unique `id` primary key between HF Hub audio and CSV text. `id` is unique across all 39,332 cleaned rows, whereas ~59% of rows share a duplicated transcript — so text-based joining cannot identify which audio clip belongs to which row and is not used.
   - **Memory Optimization:** Uses `writer_batch_size=500` during `dataset.map()` to flush Arrow cache to disk, preventing RAM OOM on large datasets.

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
│   ├── train.csv         # Training dataset (85%) — text labels only
│   └── val.csv           # Validation dataset (15%) — text labels only
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

### 2. (Recommended) Set HuggingFace Token for Faster Downloads

```bash
export HF_TOKEN="your_huggingface_token_here"
```
> The training pipeline downloads audio data from HuggingFace Hub (`shekar-ai/neyshekar-v4-persian-asr-fa`). Setting `HF_TOKEN` enables higher rate limits and faster downloads.

---

## 🚀 Execution Guide

### Step 1: Preprocess Data (Task 1)

If you wish to re-generate the cleaned datasets from raw Parquet files:

```bash
# Point this at the directory holding the 15 raw train-*.parquet files
export NEYSHEKAR_RAW_DIR="/path/to/neyshekar dataset"    # Windows: set NEYSHEKAR_RAW_DIR=E:\neyshekar dataset

python data_prep.py
```
> **Output:** Generates `data/train.csv` (85%) and `data/val.csv` (15%).
>
> The committed CSVs are already the cleaned output (33,432 train + 5,900 val = 39,332 rows from
> 40,008 raw). Re-running is only needed if you change the cleaning rules.

---

### Step 2: Quick Smoke Test (30 Steps Verification)

To verify model initialization, CUDA memory management, and metric calculation prior to full training:

```bash
# Colab-friendly: downloads only 2 of 15 shards (~1 GB instead of ~7.3 GB)
python train.py --max_steps 30 --max_shards 2 --output_dir ./whisper-large-v3-smoke-test
```

`--max_shards N` limits the download to the first N parquet shards. Without it, a 1,000-sample
smoke test still pulls the full ~7.3 GB before discarding ~97% of it. Two shards yield ~4,500 train
and ~816 validation rows — plenty to exercise the pipeline.

The smoke test intentionally exercises the paths most likely to hide a bug. Confirm you see:

| Log line | Confirms |
| :--- | :--- |
| `[MATCH][PARTIAL] ... matched` | id-join worked (partial is expected here) |
| `[VERIFY] Label alignment OK` | each transcript is attached to its own audio row |
| `decoder_start_token_id (<\|startoftranscript\|>) = 50258` | SOT strip uses the correct token |
| `[GENERATION CONFIG] Pinned language='persian'` | eval decodes Persian, not auto-detected |
| `[SMOKE TEST] Auto-scaled eval_steps=...` | eval/WER/CER/checkpoint actually run |
| an `eval_wer` / `eval_cer` value + `checkpoint-*` dir | metrics and best-checkpoint saving work |

> `--max_shards` is for smoke tests only. On a full run it prints a warning, and results from a
> partial download must never be reported.

---

### Step 3: Full Fine-Tuning (Full Training Run)

To run complete fine-tuning on all training epochs:

```bash
python train.py --max_steps -1 --epochs 3 --final_full_eval \
    --output_dir ./whisper-large-v3-neyshekar-qlora
```

> **Evaluation cost:** 3 epochs is ~6,267 optimizer steps. Every evaluation runs autoregressive
> `generate()` over the validation split, so evaluating all 5,900 rows at `eval_steps=100` would add
> roughly 4–8 hours of pure eval overhead. Defaults are therefore `eval_steps=500` with a
> deterministic 1,500-row eval subset (~12 curve points for the Task 3 analysis), and
> `--final_full_eval` computes the headline WER/CER once on the full split at the end.

> **Note:** On the first run, the Neyshekar dataset will be downloaded from HuggingFace Hub (~3-5 GB). Subsequent runs will use the cached copy.

---

## ⚙️ CLI Arguments (`train.py`)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--max_steps` | `-1` | Maximum training steps (`-1` for full training across all epochs) |
| `--epochs` | `3` | Total number of training epochs |
| `--output_dir` | `./whisper-large-v3-neyshekar-qlora` | Directory to save checkpoints and final model weights |
| `--train_csv` | `data/train.csv` | Path to training CSV dataset |
| `--val_csv` | `data/val.csv` | Path to validation CSV dataset |
| `--eval_steps` | auto | Evaluation frequency. Auto-scales down during smoke tests so the eval/WER/CER/checkpoint path is actually exercised |
| `--save_steps` | auto | Checkpoint frequency (kept a multiple of `eval_steps`) |
| `--max_eval_samples` | `1500` | Validation rows used for periodic evals. `-1` = always use the full split |
| `--max_shards` | `None` (all 15) | Download only the first N parquet shards (~485 MB each). Smoke tests only |
| `--no_quantization` | off | Load the base model in bf16 instead of 4-bit NF4. Removes dequantization overhead and lets LoRA adapt exact weights. Needs ~2.2 GB more VRAM |
| `--no_gradient_checkpointing` | off | Faster but much higher VRAM. **Measured to OOM on a 24 GB L4** |
| `--train_batch_size` / `--grad_accum` | `8` / `2` | Keep the product at 16 to preserve the learning-rate recipe |
| `--eval_batch_size` | `8` | Affects evaluation speed only, never WER/CER |
| `--no_vad_trim` / `--no_peak_norm` | off | Disable the Task 1 audio preprocessing |
| `--final_full_eval` | off | After training, evaluate once on the **full** validation set and write `final_eval_metrics.json` |

---

## 📊 Training Metrics & Logging

Training metrics, loss curves, WER, and CER evaluation logs are saved under `./logs`.  
To launch **TensorBoard**:

```bash
tensorboard --logdir ./logs
```

---

## 🛡️ VRAM & Memory Efficiency

- **Hardware Requirement:** Minimum **12 GB GPU VRAM** (e.g., RTX 3060, RTX 4070, T4, L4, V100, A100) and **32+ GB System RAM**.
- **Default Batch Config:** `PER_DEVICE_TRAIN_BATCH_SIZE = 8` with `GRADIENT_ACCUMULATION_STEPS = 2` (effective batch size = 16).
- **Gradient Checkpointing & Mixed Precision:** `FP16 = True` with gradient checkpointing enabled for peak memory optimization.
- **Dataset Processing:** Uses `writer_batch_size=500` during feature extraction to flush Arrow cache to disk, preventing RAM OOM.

---
Developed for Persian Speech Recognition Assessment (Neyshekar ASR).
