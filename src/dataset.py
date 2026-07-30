"""
Dataset Loading, Feature Extraction, and Data Collator for Whisper Fine-tuning.
Task 2 - Step 3: PyTorch Data Collator, Single-process Colab Stability, and Strict Column Sanitization.
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
    from datasets import Dataset, DatasetDict
except ImportError:
    WhisperProcessor = None
    Dataset = None
    DatasetDict = None


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
    Preprocesses a single sample/batch for Whisper model:
    - Reads audio signal (at 16 kHz sampling rate).
    - Extracts 128-channel Log-Mel spectrogram into 'input_features'.
    - Tokenizes clean text transcript into 'labels'.
    - Returns ONLY 'input_features' and 'labels'.
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

    # 3. Fallback to zero audio signal if audio binary is missing
    if audio_array is None:
        dur_sec = float(duration) if duration and not pd.isna(duration) else 3.0
        dur_sec = max(1.0, min(30.0, dur_sec))
        audio_array = np.zeros(int(16000 * dur_sec), dtype=np.float32)
        sampling_rate = 16000

    # Convert to mono if stereo
    if isinstance(audio_array, np.ndarray) and audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

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
    val_csv: str = config.VAL_CSV
) -> DatasetDict:
    """
    Loads train and val CSV files and converts them to HuggingFace DatasetDict.
    """
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train CSV not found at: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation CSV not found at: {val_csv}")

    print(f"[DATASET LOAD] Loading Train CSV from {train_csv}...")
    train_df = pd.read_csv(train_csv)
    print(f"[DATASET LOAD] Loading Val CSV from {val_csv}...")
    val_df = pd.read_csv(val_csv)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

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
    Guarantees datasets contain ONLY 'input_features' and 'labels'.
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

    # Single-process mapping by default (num_proc=1) to prevent Colab shared memory IPC EOFError
    map_kwargs = {
        "fn_kwargs": {"processor": processor},
        "remove_columns": dataset_dict["train"].column_names,
        "desc": "Preparing Features"
    }
    if num_proc is not None and num_proc > 1:
        map_kwargs["num_proc"] = num_proc

    print(f"[DATASET MAP] Processing train_dataset with prepare_dataset (num_proc={num_proc or 1})...")
    train_mapped = dataset_dict["train"].map(prepare_dataset, **map_kwargs)

    print(f"[DATASET MAP] Processing val_dataset with prepare_dataset (num_proc={num_proc or 1})...")
    val_mapped = dataset_dict["validation"].map(prepare_dataset, **map_kwargs)

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
