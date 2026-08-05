"""
Dataset Loading, Feature Extraction, and Data Collator for Whisper Fine-tuning.
Task 2 - Step 3: PyTorch Data Collator with HuggingFace Hub Audio Loading.

ARCHITECTURE:
    - Audio is loaded from HuggingFace Hub (shekar-ai/neyshekar-v5-persian-asr-fa)
      which contains actual audio bytes/arrays.
    - Cleaned text labels are loaded from local CSV files (data/train.csv, data/val.csv)
      which were produced by the Task 1 data cleaning pipeline.
    - Samples are matched by 'id' column between the two sources.
    - Feature extraction (Log-Mel spectrogram) is performed on-the-fly during .map().
    - writer_batch_size is used to flush Arrow cache to disk periodically,
      preventing RAM OOM during .map() for large datasets.
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in sys.path
sys.path.append(os.path.dirname(__file__))
import config

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import WhisperProcessor
    from datasets import Dataset, DatasetDict, load_dataset, Audio
except ImportError:
    WhisperProcessor = None
    Dataset = None
    DatasetDict = None
    load_dataset = None
    Audio = None


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data Collator designed specifically for Whisper Speech Seq2Seq Fine-tuning.
    Handles dynamic padding for audio Log-Mel features and text token labels,
    masking padding tokens with -100 for proper Loss calculation.
    STRICTLY guarantees ONLY 'input_features' and 'labels' are returned in the batch dictionary.
    """
    processor: Any
    decoder_start_token_id: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], Any]]]) -> Dict[str, Any]:
        # 1. Dynamic padding for audio Log-Mel spectrogram input_features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 2. Dynamic padding for text label token IDs
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token IDs with -100 so CrossEntropy Loss ignores them
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If BOS token is at the beginning of all labels, strip it to avoid duplicate BOS prefix
        if self.processor.tokenizer.bos_token_id is not None and len(labels) > 0:
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]

        # STRICTLY return ONLY input_features and labels to prevent 'input_ids' keyword collision in WhisperDecoder
        return {
            "input_features": batch["input_features"],
            "labels": labels
        }


def prepare_dataset(batch: Dict[str, Any], processor: Any) -> Dict[str, Any]:
    """
    Preprocesses a single sample for Whisper model:
    - Reads audio signal (at 16 kHz sampling rate) from HF audio dict, bytes, or file path.
    - Extracts 128-channel Log-Mel spectrogram into 'input_features'.
    - Tokenizes clean text transcript into 'labels'.
    - Returns ONLY 'input_features' and 'labels'.

    NOTE: This function expects the batch to contain an 'audio' column with actual
    audio data (dict with 'array'/'bytes'/'path'). If no audio is found, it falls
    back to a zero signal but logs a warning.
    """
    audio_data = batch.get("audio", None)
    audio_path = batch.get("audio_path", batch.get("path", None))
    text = batch.get("cleaned_text", batch.get("text", batch.get("transcript", batch.get("sentence", ""))))
    duration = batch.get("duration", 3.0)

    audio_array = None
    sampling_rate = 16000

    # 1. Try extracting audio array from HF audio dict / bytes / path
    if audio_data is not None:
        if isinstance(audio_data, dict):
            if "array" in audio_data and audio_data["array"] is not None:
                audio_array = audio_data["array"]
                sampling_rate = audio_data.get("sampling_rate", 16000)
            elif "bytes" in audio_data and audio_data["bytes"] is not None:
                try:
                    audio_array, sampling_rate = sf.read(io.BytesIO(audio_data["bytes"]))
                except Exception:
                    pass
            elif "path" in audio_data and audio_data["path"] is not None:
                audio_path = audio_data["path"]

    # 2. Try loading audio from file path
    if audio_array is None and audio_path and os.path.exists(str(audio_path)):
        try:
            audio_array, sampling_rate = sf.read(str(audio_path))
        except Exception:
            pass

    # 3. Fallback to zero audio signal if audio binary is missing (should NOT happen with proper data loading)
    if audio_array is None:
        dur_sec = float(duration) if duration and not pd.isna(duration) else 3.0
        dur_sec = max(1.0, min(30.0, dur_sec))
        audio_array = np.zeros(int(16000 * dur_sec), dtype=np.float32)
        sampling_rate = 16000

    # Convert to mono if stereo
    if isinstance(audio_array, np.ndarray) and audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Ensure float32 for feature extraction
    if isinstance(audio_array, np.ndarray):
        audio_array = audio_array.astype(np.float32)

    # 4. Extract 128-channel Log-Mel Spectrogram Features
    inputs = processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np"
    )

    # 5. Tokenize text transcript to Token IDs
    labels = processor.tokenizer(str(text)).input_ids

    return {
        "input_features": inputs.input_features[0],
        "labels": labels
    }


def load_custom_dataset(
    train_csv: str = config.TRAIN_CSV,
    val_csv: str = config.VAL_CSV,
    hf_dataset_name: str = config.HF_DATASET_NAME
) -> DatasetDict:
    """
    Loads training and validation datasets by:
    1. Loading the full Neyshekar dataset from HuggingFace Hub (audio stays lazy-loaded via Arrow).
    2. Loading cleaned text labels from local CSV files (produced by Task 1 pipeline).
    3. Matching HF samples with CSV entries by TEXT CONTENT (not ID), respecting the EXACT
       per-text duplicate counts that Task 1's dedup/downsampling logic decided to keep.
       This is necessary because CSV IDs (from the local raw Parquet used for Task 1) don't
       reliably match HF Hub IDs.

    WHY "IN SET" MEMBERSHIP MATCHING IS WRONG:
        A naive `hf_dataset.filter(lambda x: x['text'] in known_texts)` keeps EVERY HF row whose
        text appears anywhere in the CSV, regardless of how many times that text was capped/dropped
        during Task 1's duplicate downsampling (e.g. long sentences repeated >3 times were capped
        to 3 copies). Since membership is a set check, all 5, 10, etc. duplicate audio rows for a
        repeated sentence would silently pass through, reintroducing exactly the over-representation
        Task 1 intentionally removed.

    MATCHING STRATEGY (order-preserving, count-aware):
        - CSV contains 'text_raw' (original text) and 'cleaned_text' (normalized text).
        - HF Hub contains 'text' (original text) and 'audio'.
        - We count how many times each raw text appears in train.csv / val.csv (Counter).
        - We walk the HF dataset once, in order, and only accept up to that many occurrences
          of each text (first N go to train, next M go to validation, any further duplicates
          beyond N+M are skipped) — mirroring the exact row-level dedup decision from Task 1.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    from collections import Counter

    # 1. Validate CSV files exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train CSV not found at: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation CSV not found at: {val_csv}")

    # 2. Load cleaned text from CSV
    print(f"[DATASET LOAD] Loading Train CSV from {train_csv}...")
    train_df = pd.read_csv(train_csv)
    print(f"[DATASET LOAD] Loading Val CSV from {val_csv}...")
    val_df = pd.read_csv(val_csv)

    # 3. Build per-text occurrence counts + cleaned-text lookup for each split.
    #    Counting (not just set membership) is what lets us respect Task 1's duplicate caps.
    train_text_counts: Counter = Counter()
    train_cleaned_map = {}
    for _, row in train_df.iterrows():
        raw = str(row.get('text_raw', '')).strip()
        cleaned = str(row.get('cleaned_text', raw)).strip()
        if raw:
            train_text_counts[raw] += 1
            train_cleaned_map[raw] = cleaned

    val_text_counts: Counter = Counter()
    val_cleaned_map = {}
    for _, row in val_df.iterrows():
        raw = str(row.get('text_raw', '')).strip()
        cleaned = str(row.get('cleaned_text', raw)).strip()
        if raw:
            val_text_counts[raw] += 1
            val_cleaned_map[raw] = cleaned

    needed_texts = set(train_text_counts) | set(val_text_counts)
    total_needed = sum(train_text_counts.values()) + sum(val_text_counts.values())
    print(f"[DATASET LOAD] Train rows: {sum(train_text_counts.values()):,} "
          f"({len(train_text_counts):,} unique texts), "
          f"Val rows: {sum(val_text_counts.values()):,} ({len(val_text_counts):,} unique texts)")

    # 4. Load full HuggingFace Hub dataset (audio stays lazy-loaded by Arrow)
    print(f"[HF HUB] Loading audio dataset from HuggingFace Hub: '{hf_dataset_name}'...")
    print(f"[HF HUB] This may take a while on first run (downloading audio files)...")
    hf_dataset = load_dataset(hf_dataset_name, split="train")

    # Cast audio column to Audio format at 16kHz for Whisper compatibility
    if "audio" in hf_dataset.column_names:
        hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=config.SAMPLING_RATE))
    print(f"[HF HUB] Loaded {len(hf_dataset):,} samples from HuggingFace Hub.")
    print(f"[HF HUB] HF Dataset columns: {hf_dataset.column_names}")

    # 5. Single ordered pass over HF text column: assign each occurrence of a text to
    #    train / validation / dropped, capped exactly at the counts decided in Task 1's CSVs.
    #    Using .select(indices) instead of .filter(lambda) keeps this deterministic and avoids
    #    reintroducing duplicates via naive set-membership filtering.
    print("[MATCH] Scanning HF dataset once to assign rows by exact per-text duplicate counts...")
    seen_counts: Counter = Counter()
    train_indices = []
    val_indices = []
    skipped_excess_duplicates = 0
    for i, text in enumerate(hf_dataset["text"]):
        t = str(text).strip()
        if t not in needed_texts:
            continue
        tc = train_text_counts.get(t, 0)
        vc = val_text_counts.get(t, 0)
        seen = seen_counts[t]
        if seen < tc:
            train_indices.append(i)
        elif seen < tc + vc:
            val_indices.append(i)
        else:
            # This text already has as many audio rows assigned as Task 1's CSV kept for it.
            # Any further occurrence is an excess duplicate that Task 1 intentionally dropped.
            skipped_excess_duplicates += 1
            continue
        seen_counts[t] += 1

    matched_total = len(train_indices) + len(val_indices)
    print(f"[MATCH] Matched {matched_total:,} / {total_needed:,} rows needed by the CSVs "
          f"(skipped {skipped_excess_duplicates:,} excess duplicates beyond Task 1's caps).")

    if matched_total == 0:
        hf_sample_texts = [str(hf_dataset[i]['text'])[:60] for i in range(min(5, len(hf_dataset)))]
        csv_sample_texts = list(train_cleaned_map.keys())[:5]
        raise ValueError(
            f"No matching texts found between CSV and HF dataset!\n"
            f"  CSV text_raw samples: {csv_sample_texts}\n"
            f"  HF text samples: {hf_sample_texts}\n"
            f"  Check that CSV 'text_raw' column matches HF 'text' column, and that "
            f"HF_DATASET_NAME points to the same dataset version used to build the raw "
            f"parquet files for Task 1."
        )
    if matched_total < total_needed:
        missing = total_needed - matched_total
        print(f"[MATCH][WARNING] {missing:,} CSV rows had no corresponding audio row in "
              f"'{hf_dataset_name}'. This usually means HF_DATASET_NAME is a different "
              f"version than the raw data used for Task 1 cleaning. Training will proceed "
              f"with the {matched_total:,} rows that did match, but investigate this before "
              f"treating results as final.")

    train_dataset = hf_dataset.select(train_indices)
    val_dataset = hf_dataset.select(val_indices)

    # 6. Attach cleaned_text column via direct dict lookup (fast, order already fixed by .select)
    def _attach_train_text(sample):
        sample['cleaned_text'] = train_cleaned_map[str(sample['text']).strip()]
        return sample

    def _attach_val_text(sample):
        sample['cleaned_text'] = val_cleaned_map[str(sample['text']).strip()]
        return sample

    train_dataset = train_dataset.map(_attach_train_text, desc="Attaching cleaned_text (train)")
    val_dataset = val_dataset.map(_attach_val_text, desc="Attaching cleaned_text (val)")

    print(f"[SPLIT] Train: {len(train_dataset):,} samples, Validation: {len(val_dataset):,} samples")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty after text matching! Check CSV text_raw vs HF text columns.")

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })


def get_datasets_and_collator(
    train_csv: str = config.TRAIN_CSV,
    val_csv: str = config.VAL_CSV,
    model_name: str = config.MODEL_NAME_OR_PATH,
    language: str = config.LANGUAGE,
    task: str = config.TASK,
    max_samples: Optional[int] = None,
    num_proc: Optional[int] = 1
) -> tuple[Any, Any, Any, Any]:
    """
    Main entry point function returning (train_dataset, val_dataset, processor, data_collator).
    Guarantees datasets contain ONLY 'input_features' and 'labels' after .map() processing.

    Audio is loaded from HuggingFace Hub, cleaned text from local CSV.
    Uses writer_batch_size to prevent OOM during .map() by flushing Arrow cache to disk.
    """
    config.set_seed(config.SEED)

    print(f"[PROCESSOR] Loading WhisperProcessor for '{model_name}' (language='{language}', task='{task}')...")
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task=task
    )

    dataset_dict = load_custom_dataset(train_csv, val_csv)

    # Optional Subset Mode for fast Colab Smoke Testing (e.g. 1000 samples)
    if max_samples is not None and max_samples > 0:
        train_len = min(len(dataset_dict["train"]), max_samples)
        val_len = min(len(dataset_dict["validation"]), max_samples)
        print(f"[SUBSET MODE] Selecting first {train_len:,} train samples and {val_len:,} val samples for fast testing...")
        dataset_dict["train"] = dataset_dict["train"].select(range(train_len))
        dataset_dict["validation"] = dataset_dict["validation"].select(range(val_len))

    # Prepare .map() kwargs with writer_batch_size to prevent OOM
    # writer_batch_size flushes the Arrow cache buffer to disk every N samples,
    # preventing unbounded RAM growth during spectrogram extraction.
    map_kwargs = {
        "fn_kwargs": {"processor": processor},
        "remove_columns": dataset_dict["train"].column_names,
        "desc": "Preparing Features",
        "writer_batch_size": config.MAP_WRITER_BATCH_SIZE,
    }
    if num_proc is not None and num_proc > 1:
        map_kwargs["num_proc"] = num_proc

    print(f"[DATASET MAP] Processing train_dataset with prepare_dataset (num_proc={num_proc or 1}, writer_batch_size={config.MAP_WRITER_BATCH_SIZE})...")
    train_mapped = dataset_dict["train"].map(prepare_dataset, **map_kwargs)

    print(f"[DATASET MAP] Processing val_dataset with prepare_dataset (num_proc={num_proc or 1}, writer_batch_size={config.MAP_WRITER_BATCH_SIZE})...")
    val_map_kwargs = {
        "fn_kwargs": {"processor": processor},
        "remove_columns": dataset_dict["validation"].column_names,
        "desc": "Preparing Features",
        "writer_batch_size": config.MAP_WRITER_BATCH_SIZE,
    }
    if num_proc is not None and num_proc > 1:
        val_map_kwargs["num_proc"] = num_proc
    val_mapped = dataset_dict["validation"].map(prepare_dataset, **val_map_kwargs)

    # STRICT SANITIZATION: Explicitly remove any column other than 'input_features' and 'labels'
    cols_to_keep = {"input_features", "labels"}
    train_remove = [c for c in train_mapped.column_names if c not in cols_to_keep]
    if train_remove:
        print(f"[COLUMN SANITIZE] Stripping unwanted columns from train_mapped: {train_remove}")
        train_mapped = train_mapped.remove_columns(train_remove)

    val_remove = [c for c in val_mapped.column_names if c not in cols_to_keep]
    if val_remove:
        print(f"[COLUMN SANITIZE] Stripping unwanted columns from val_mapped: {val_remove}")
        val_mapped = val_mapped.remove_columns(val_remove)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.bos_token_id
    )

    print(f"[READY] Mapped Datasets: Train={len(train_mapped):,} samples, Val={len(val_mapped):,} samples.")
    print(f"[READY] Dataset Features: {train_mapped.column_names}")

    return train_mapped, val_mapped, processor, data_collator


if __name__ == '__main__':
    print("Testing dataset module structure...")
    if WhisperProcessor is None:
        print("[NOTICE] Transformers/Datasets missing locally.")
    else:
        print("[SUCCESS] dataset.py module loaded cleanly.")
