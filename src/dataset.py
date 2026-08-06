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

        # 3. Strip the leading <|startoftranscript|> (SOT) token if present.
        #
        # CRITICAL: this MUST compare against decoder_start_token_id (50258 = <|startoftranscript|>),
        # NOT tokenizer.bos_token_id. For Whisper, bos_token is "<|endoftext|>" (id 50257), which is
        # NOT the token the tokenizer actually places at position 0 of a label sequence. A tokenized
        # Persian label begins:
        #     [50258 <|startoftranscript|>, 50300 <|fa|>, 50360 <|transcribe|>, 50364 <|notimestamps|>, ...]
        # so a check against bos_token_id (50257) can never match and the SOT token is never removed.
        # Whisper's shift_tokens_right() then prepends decoder_start_token_id again, producing a
        # duplicated "<|sot|> <|sot|> <|fa|> <|transcribe|>" decoder prefix that silently corrupts
        # teacher forcing and inflates WER/CER without ever raising an error.
        if self.decoder_start_token_id is not None and len(labels) > 0:
            if (labels[:, 0] == self.decoder_start_token_id).all():
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
    duration = batch.get("duration", 3.0)

    # The label MUST come from 'cleaned_text' -- the Task 1 output (Arabic->Persian normalised,
    # digits lexicalised). The previous code fell back to the Hub's raw 'text' column whenever
    # 'cleaned_text' was absent, which would have silently trained the model on UNCLEANED
    # transcripts (Arabic characters, bare digits) with no error and no warning. Since the whole
    # point of Task 1 is that the model sees cleaned text, that fallback is never acceptable:
    # a missing label means the id-join or the cleaned_text attachment broke, and we want to know.
    if "cleaned_text" not in batch:
        raise RuntimeError(
            "Sample has no 'cleaned_text' column -- refusing to fall back to the raw 'text' "
            "column, which would silently train on UNCLEANED transcripts.\n"
            f"  available columns: {sorted(batch.keys())}\n"
            "This means load_custom_dataset() failed to attach the Task 1 labels."
        )
    text = batch["cleaned_text"]
    if not str(text).strip():
        raise RuntimeError(
            f"Empty 'cleaned_text' label for a sample (id={batch.get('id')!r}). "
            "Task 1 guarantees no empty labels, so this indicates data corruption."
        )

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

    # 1b. datasets >= 3.x / 4.x returns a torchcodec `AudioDecoder` object for decoded audio
    #     columns instead of the classic {"array", "sampling_rate"} dict. Without this branch the
    #     object matches none of the cases above and the sample looks like "missing audio".
    #     NOTE: torchcodec returns CHANNEL-FIRST data of shape [num_channels, num_samples],
    #     the opposite of soundfile's [num_samples, num_channels] -- so mono-downmix here must
    #     average over axis 0, and we return a 1-D array so the generic downmix below is a no-op.
    if audio_array is None and hasattr(audio_data, "get_all_samples"):
        try:
            samples = audio_data.get_all_samples()
            data = samples.data
            data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
            if data.ndim > 1:
                data = data.mean(axis=0)
            audio_array = data
            sampling_rate = int(samples.sample_rate)
        except Exception:
            pass

    # 1c. Some versions expose plain `.array` / `.sampling_rate` attributes instead.
    if audio_array is None and hasattr(audio_data, "array"):
        try:
            audio_array = np.asarray(audio_data.array)
            sampling_rate = int(getattr(audio_data, "sampling_rate", 16000))
        except Exception:
            pass

    # 2. Try loading audio from file path
    if audio_array is None and audio_path and os.path.exists(str(audio_path)):
        try:
            audio_array, sampling_rate = sf.read(str(audio_path))
        except Exception:
            pass

    # 3. FAIL LOUDLY if audio could not be decoded.
    #
    # This used to silently substitute a zero (pure silence) waveform. That is dangerous: training
    # would appear to run perfectly while the model was being taught to map SILENCE -> real Persian
    # transcripts. That is precisely the "hallucination on silence" failure mode documented in the
    # Task 1 report, and it produces no error, no warning, and no obviously wrong loss curve --
    # it would only surface as inexplicably bad WER after burning a full training run.
    # Given the cost of a full fine-tune, crashing early with a clear message is strictly better.
    if audio_array is None:
        raise RuntimeError(
            "Failed to decode audio for a sample (no usable 'array', 'bytes', or readable 'path').\n"
            f"  transcript preview : {str(text)[:80]!r}\n"
            f"  audio field type   : {type(audio_data).__name__}\n"
            f"  audio_path         : {audio_path!r}\n"
            "Refusing to substitute a zero/silent waveform, because that would silently train the "
            "model to transcribe silence into text. Check that the HuggingFace dataset was fully "
            "downloaded and that the 'audio' column was cast with Audio(sampling_rate=16000)."
        )

    # 4. Normalize waveform shape/dtype: force ndarray, mono, float32.
    audio_array = np.asarray(audio_array)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)

    # 5. Resample to 16 kHz if the source rate differs.
    # Whisper's feature extractor assumes 16 kHz; feeding it e.g. 44.1 kHz audio while *claiming*
    # sampling_rate=16000 silently time-stretches the signal and destroys audio/text alignment.
    if sampling_rate != 16000:
        duration_sec = len(audio_array) / float(sampling_rate)
        target_len = int(round(duration_sec * 16000))
        if target_len > 0:
            audio_array = np.interp(
                np.linspace(0.0, len(audio_array) - 1, target_len),
                np.arange(len(audio_array)),
                audio_array,
            ).astype(np.float32)

    # 6. Extract 128-channel Log-Mel Spectrogram Features
    inputs = processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np"
    )

    # 7. Tokenize text transcript to Token IDs.
    # Truncate to Whisper's decoder limit (max_target_positions = 448). Whisper has learned
    # absolute positional embeddings for only 448 decoder positions, so a longer label triggers
    # an index-out-of-range CUDA assert mid-training rather than a clean error.
    labels = processor.tokenizer(
        str(text),
        truncation=True,
        max_length=448,
    ).input_ids

    return {
        "input_features": inputs.input_features[0],
        "labels": labels
    }


def load_custom_dataset(
    train_csv: str = config.TRAIN_CSV,
    val_csv: str = config.VAL_CSV,
    hf_dataset_name: str = config.HF_DATASET_NAME,
    max_shards: Optional[int] = None
) -> DatasetDict:
    """
    Loads training and validation datasets by:
    1. Loading the full Neyshekar dataset from HuggingFace Hub (audio stays lazy-loaded via Arrow).
    2. Loading cleaned text labels from local CSV files (produced by Task 1 pipeline).
    3. Matching HF rows to CSV rows by the stable 'id' primary key.

    WHY 'id' AND NOT TEXT:
        Both the CSVs and the Hub dataset descend from the SAME source (neyshekar v4, 40,008 rows
        with an int64 'id'). Task 1 preserved that 'id' column, and it is unique across all 39,332
        cleaned rows (verified: 39,332 rows -> 39,332 distinct ids, range 0..40,007). So 'id' is an
        exact, order-independent primary key.

        Transcript text is NOT a usable key here: ~10,500 distinct transcripts occur more than once
        (about 59% of all rows are involved, one sentence appears 11 times). Joining on text
        therefore cannot tell which specific audio clip belongs to which row, and it silently pairs
        transcripts with arbitrary audio of the same sentence. Two earlier variants of this function
        both got this wrong:
          - `filter(lambda x: x['text'] in known_texts)` -- a set-membership test that re-admitted
            every duplicate Task 1 had deliberately dropped, undoing the deduplication.
          - a count-capped text walk -- respected the per-text counts but still assigned audio to
            transcripts arbitrarily, and could place a clip in train whose Task 1 row was in val,
            leaking data across the split boundary.
        Matching on 'id' removes the ambiguity entirely and reproduces Task 1's split exactly.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
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

    for name, df in (("train", train_df), ("validation", val_df)):
        missing_cols = {"id", "cleaned_text"} - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"{name} CSV is missing required column(s) {sorted(missing_cols)}. "
                f"Found columns: {list(df.columns)}. Re-run data_prep.py to regenerate the CSVs."
            )

    # 3. Build id -> cleaned_text lookups for each split.
    train_id_map = {int(r.id): str(r.cleaned_text) for r in train_df.itertuples(index=False)}
    val_id_map = {int(r.id): str(r.cleaned_text) for r in val_df.itertuples(index=False)}

    overlap = set(train_id_map) & set(val_id_map)
    if overlap:
        raise ValueError(
            f"TRAIN/VAL LEAKAGE: {len(overlap):,} id(s) appear in BOTH train.csv and val.csv "
            f"(e.g. {sorted(overlap)[:5]}). Every reported validation metric would be optimistic. "
            f"Re-run data_prep.py to regenerate a clean split."
        )

    total_needed = len(train_id_map) + len(val_id_map)
    print(f"[DATASET LOAD] Train rows: {len(train_id_map):,}, Val rows: {len(val_id_map):,} "
          f"(total {total_needed:,}, matched by unique 'id')")

    # 4. Load the HuggingFace Hub audio dataset (audio stays lazy-loaded by Arrow).
    #
    # SMOKE-TEST ECONOMY: the full v4 dataset is ~7.3 GB spread over 15 parquet shards
    # (~485 MB / ~2,667 rows each). Subsetting after load_dataset() would still download all
    # 7.3 GB before discarding ~97% of it -- painful on Colab, where the disk and the session
    # are both disposable. Passing max_shards downloads only the first N shards instead.
    print(f"[HF HUB] Loading audio dataset from HuggingFace Hub: '{hf_dataset_name}'...")
    if max_shards is not None and max_shards > 0:
        shard_files = [f"data/train-{i:05d}-of-00015.parquet" for i in range(max_shards)]
        print(f"[HF HUB][PARTIAL] Downloading only {max_shards} of 15 shards "
              f"(~{max_shards * 485} MB instead of ~7.3 GB): {shard_files}")
        # verification_mode="no_checks" is REQUIRED here. The dataset card declares a split of
        # 40,008 examples / 7.17 GB, and by default `datasets` verifies the loaded split against
        # that metadata and raises NonMatchingSplitsSizesError. Loading a subset of shards is
        # deliberately "wrong" by that measure, so the check must be disabled for this path only.
        hf_dataset = load_dataset(
            hf_dataset_name,
            split="train",
            data_files={"train": shard_files},
            verification_mode="no_checks",
        )
    else:
        # Full load keeps verification ON: here a size mismatch is a genuine red flag
        # (truncated or corrupted download) and should fail loudly.
        print(f"[HF HUB] This may take a while on first run (downloading ~7.3 GB of audio)...")
        hf_dataset = load_dataset(hf_dataset_name, split="train")

    # Keep the audio column UNDECODED (raw bytes), and decode it ourselves in prepare_dataset.
    #
    # WHY decode=False: the neyshekar v4 card already stores audio with decode=false. Casting to
    # Audio(sampling_rate=...) sets decode=True, which on datasets >= 3.x hands back a torchcodec
    # `AudioDecoder` object rather than the classic {"array", "sampling_rate"} dict. That silently
    # broke feature extraction. Decoding via soundfile ourselves keeps behaviour identical across
    # datasets versions, and prepare_dataset already resamples to 16 kHz when needed.
    if "audio" in hf_dataset.column_names:
        try:
            hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
            print("[HF HUB] Audio column kept undecoded (raw bytes); decoding via soundfile.")
        except Exception as e:
            print(f"[HF HUB][WARNING] Could not cast audio column to decode=False ({e}). "
                  f"Falling back to the default decoded representation.")
    print(f"[HF HUB] Loaded {len(hf_dataset):,} samples from HuggingFace Hub.")
    print(f"[HF HUB] HF Dataset columns: {hf_dataset.column_names}")

    if "id" not in hf_dataset.column_names:
        raise ValueError(
            f"HF dataset '{hf_dataset_name}' has no 'id' column (columns: {hf_dataset.column_names}). "
            f"Expected the neyshekar v4 schema: ['id', 'audio', 'text', 'duration']. "
            f"Check that HF_DATASET_NAME points at the same version used for Task 1."
        )

    # 5. Single ordered pass over the id column, assigning each HF row to train / val / unused.
    #    .select(indices) keeps this deterministic and avoids per-row lambda filtering.
    print("[MATCH] Matching HF rows to CSV rows by 'id'...")
    train_indices, val_indices = [], []
    train_ids_seen, val_ids_seen = [], []
    for i, raw_id in enumerate(hf_dataset["id"]):
        rid = int(raw_id)
        if rid in train_id_map:
            train_indices.append(i)
            train_ids_seen.append(rid)
        elif rid in val_id_map:
            val_indices.append(i)
            val_ids_seen.append(rid)

    matched_total = len(train_indices) + len(val_indices)
    print(f"[MATCH] Matched {matched_total:,} / {total_needed:,} CSV rows to audio "
          f"(train={len(train_indices):,}, val={len(val_indices):,}).")

    if matched_total == 0:
        raise ValueError(
            f"No ids matched between the CSVs and '{hf_dataset_name}'!\n"
            f"  CSV id sample : {sorted(train_id_map)[:5]}\n"
            f"  HF  id sample : {[int(x) for x in hf_dataset['id'][:5]]}\n"
            f"  The CSVs were probably generated from a different dataset version. "
            f"Re-run data_prep.py against the same version as HF_DATASET_NAME."
        )
    if matched_total < total_needed:
        if max_shards is not None and max_shards > 0:
            # Expected: we deliberately downloaded only part of the corpus, so most CSV rows have
            # no audio available. Not an error -- but never acceptable for a reportable run.
            print(f"[MATCH][PARTIAL] {matched_total:,} of {total_needed:,} CSV rows matched, because "
                  f"only {max_shards}/15 shards were downloaded. This is expected for a smoke test. "
                  f"Do NOT use this mode for the real 3-epoch run or for reported metrics.")
        else:
            raise ValueError(
                f"Only {matched_total:,} of {total_needed:,} CSV rows found a matching audio row in "
                f"'{hf_dataset_name}' ({total_needed - matched_total:,} missing). Training on a silently "
                f"truncated dataset would invalidate the Task 3 analysis, so this is treated as fatal. "
                f"Verify that HF_DATASET_NAME matches the raw parquet files used for Task 1 cleaning."
            )

    train_dataset = hf_dataset.select(train_indices)
    val_dataset = hf_dataset.select(val_indices)

    # 6. Attach the Task 1 cleaned transcript, keyed by the same 'id'.
    #    Index-aligned with the *_ids_seen lists captured during the select pass above, so the
    #    label attached to each row is guaranteed to be that row's own cleaned transcript.
    def _attach(dataset, ids_seen, id_map, desc):
        def _fn(sample, idx):
            sample["cleaned_text"] = id_map[ids_seen[idx]]
            return sample
        return dataset.map(_fn, with_indices=True, desc=desc)

    train_dataset = _attach(train_dataset, train_ids_seen, train_id_map, "Attaching cleaned_text (train)")
    val_dataset = _attach(val_dataset, val_ids_seen, val_id_map, "Attaching cleaned_text (val)")

    # 7. Sanity check: confirm labels really did land on the right rows.
    if len(train_dataset) > 0:
        probe = train_dataset[0]
        expected = train_id_map[int(probe["id"])]
        if str(probe["cleaned_text"]) != expected:
            raise RuntimeError(
                "Label alignment check FAILED: cleaned_text does not correspond to the row's id.\n"
                f"  id={probe['id']}\n  attached={probe['cleaned_text']!r}\n  expected={expected!r}"
            )
        print(f"[VERIFY] Label alignment OK (probed id={probe['id']}).")

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
    num_proc: Optional[int] = 1,
    max_shards: Optional[int] = None
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

    dataset_dict = load_custom_dataset(train_csv, val_csv, max_shards=max_shards)

    # Optional Subset Mode for fast Colab Smoke Testing (e.g. 1000 samples).
    # Rows are shuffled with the global SEED before selecting, rather than taking the first N.
    # The Hub stores rows in shard/collection order, so the first N rows are not a representative
    # sample (they can skew heavily toward particular speakers or recording sessions) -- a smoke
    # test on such a slice can look misleadingly good or bad. shuffle(seed) keeps it both
    # representative and exactly reproducible across runs.
    if max_samples is not None and max_samples > 0:
        train_len = min(len(dataset_dict["train"]), max_samples)
        val_len = min(len(dataset_dict["validation"]), max_samples)
        print(f"[SUBSET MODE] Selecting a random but reproducible {train_len:,} train / "
              f"{val_len:,} val samples (seed={config.SEED}) for fast testing...")
        dataset_dict["train"] = dataset_dict["train"].shuffle(seed=config.SEED).select(range(train_len))
        dataset_dict["validation"] = dataset_dict["validation"].shuffle(seed=config.SEED).select(range(val_len))

    if len(dataset_dict["validation"]) == 0:
        raise ValueError(
            "Validation split is EMPTY. With --max_shards, none of the val.csv ids happened to "
            "fall in the downloaded shards. Increase --max_shards (or drop it) so evaluation, "
            "WER/CER and best-checkpoint selection can actually run."
        )

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

    # Resolve the true decoder start token (<|startoftranscript|>, id 50258 for whisper-large-v3).
    # NOTE: deliberately NOT processor.tokenizer.bos_token_id -- for Whisper that is <|endoftext|>
    # (50257) and would never match the actual first label token, silently disabling the SOT strip.
    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    print(f"[COLLATOR] decoder_start_token_id (<|startoftranscript|>) = {decoder_start_token_id} "
          f"(tokenizer.bos_token_id = {processor.tokenizer.bos_token_id} -- intentionally not used)")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id
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
