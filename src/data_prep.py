"""
Data Preparation & Cleaning Pipeline for Neyshekar Persian ASR Dataset.
Task 2 - Step 1: Preprocessing, Deduplication, Audio Filtering, and Train/Val Split.
"""

import sys
import os
import io
import glob
import random
import hashlib
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

# Force UTF-8 encoding for standard output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path if needed
sys.path.append(os.path.dirname(__file__))
from text_cleaner import (
    apply_step1_normalization,
    remove_diacritics,
    remove_spaces_and_zwnj
)

# Configuration Constants
RANDOM_SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
MAX_COPIES_PER_LONG_TEXT = 3
DATASET_PATH = os.environ.get('NEYSHEKAR_RAW_DIR', os.path.join(os.path.dirname(__file__), '..', 'raw_data'))
OUTPUT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def set_seed(seed: int = 42) -> None:
    """Sets random seed across all relevant python libraries for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[REPRODUCIBILITY] Random seed explicitly set to {seed}")


def compute_audio_hash(audio_dict) -> str:
    """Computes MD5 hash of audio bytes if available, else path string."""
    if isinstance(audio_dict, dict):
        audio_bytes = audio_dict.get('bytes')
        if audio_bytes is not None:
            return hashlib.md5(audio_bytes).hexdigest()
        path = audio_dict.get('path')
        if path:
            return str(path)
    return str(audio_dict)


def load_raw_dataset(dataset_dir: str) -> pd.DataFrame:
    """Loads all 15 Parquet files from the raw Neyshekar dataset directory."""
    parquet_files = sorted(glob.glob(os.path.join(dataset_dir, 'train-*.parquet')))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at: {dataset_dir}")
    
    print(f"[DATA LOAD] Reading {len(parquet_files)} parquet files from {dataset_dir}...")
    dfs = [pd.read_parquet(f, columns=['id', 'audio', 'text', 'duration']) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df['text_raw'] = df['text'].fillna('').astype(str)
    return df


def preprocess_and_clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Executes complete data quality pipeline established in Task 1:
    1. Fast Audio Header Duration & Integrity Check (< 1.0s filter)
    2. Text Standardization (Arabic->Persian, Digit Lexicalization, extra space removal)
    3. Multi-Stage Deduplication (Exact row duplicate removal & Threshold-based Down-sampling)
    """
    initial_total = len(df)
    stats = {'initial_total': initial_total}
    
    # 1. Audio Integrity & Duration Extraction
    print("[PROCESSING 1/3] Extracting audio duration and checking audio health...")
    audio_durations = []
    corrupted_count = 0
    
    for idx, row in df.iterrows():
        try:
            audio_obj = row['audio']
            if not isinstance(audio_obj, dict) or 'bytes' not in audio_obj or not audio_obj['bytes']:
                corrupted_count += 1
                audio_durations.append(np.nan)
                continue
            info = sf.info(io.BytesIO(audio_obj['bytes']))
            audio_durations.append(info.duration)
        except Exception:
            corrupted_count += 1
            audio_durations.append(np.nan)
            
    df['audio_duration'] = audio_durations
    stats['corrupted_count'] = corrupted_count
    
    # Filter valid duration (>= 1.0s and <= 30.0s)
    under_1s_mask = df['audio_duration'] < 1.0
    over_30s_mask = df['audio_duration'] > 30.0
    stats['under_1s_count'] = int(under_1s_mask.sum())
    stats['over_30s_count'] = int(over_30s_mask.sum())
    
    valid_audio_mask = (df['audio_duration'] >= 1.0) & (df['audio_duration'] <= 30.0) & (~df['audio_duration'].isna())
    df_valid = df[valid_audio_mask].copy().reset_index(drop=True)
    
    # 2. Text Standardization
    print("[PROCESSING 2/3] Standardizing text (Arabic->Persian, Numbers->Words, Normalizing spaces)...")
    df_valid['normalized_text'] = df_valid['text_raw'].apply(apply_step1_normalization)
    df_valid['normalized_text'] = df_valid['normalized_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 3. Deduplication (Exact Row & Threshold Down-sampling)
    print("[PROCESSING 3/3] Running deduplication & threshold-based downsampling...")
    df_valid['audio_hash'] = df_valid['audio'].apply(compute_audio_hash)
    
    exact_dup_mask = df_valid.duplicated(subset=['audio_hash', 'text_raw'], keep='first')
    exact_dups_dropped = int(exact_dup_mask.sum())
    stats['exact_dups_dropped'] = exact_dups_dropped
    
    df_stage1 = df_valid[~exact_dup_mask].copy().reset_index(drop=True)
    
    # Create fingerprint for duplicate sentence matching
    df_stage1['text_fingerprint'] = df_stage1['normalized_text'].apply(lambda x: remove_spaces_and_zwnj(remove_diacritics(x)))
    
    rows_to_drop = []
    long_excess_dropped = 0
    short_kept = 0
    
    grouped = df_stage1.groupby('text_fingerprint')
    for fp, group_indices in grouped.groups.items():
        count = len(group_indices)
        if count > 1:
            first_text = df_stage1.loc[group_indices[0], 'normalized_text']
            is_short = (len(first_text) < 15) or (len(first_text.split()) < 4)
            if is_short:
                short_kept += count
            else:
                if count > MAX_COPIES_PER_LONG_TEXT:
                    drop_indices = group_indices[MAX_COPIES_PER_LONG_TEXT:]
                    rows_to_drop.extend(drop_indices)
                    long_excess_dropped += len(drop_indices)
                    
    stats['long_excess_dropped'] = long_excess_dropped
    df_cleaned = df_stage1.drop(index=rows_to_drop).copy().reset_index(drop=True)
    
    # Select clean columns
    df_final = df_cleaned[['id', 'audio_duration', 'text_raw', 'normalized_text']].rename(
        columns={'normalized_text': 'cleaned_text', 'audio_duration': 'duration'}
    )
    
    stats['final_clean_total'] = len(df_final)
    return df_final, stats


def prepare_and_save_data():
    """Main pipeline execution for data preparation, train/val split, and saving CSV files."""
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    # Load and clean
    df_raw = load_raw_dataset(DATASET_PATH)
    df_clean, stats = preprocess_and_clean_data(df_raw)
    
    # Split Train (85%) and Validation (15%)
    print(f"\n[TRAIN/VAL SPLIT] Splitting {len(df_clean):,} records into {TRAIN_RATIO*100:.0f}% Train and {VAL_RATIO*100:.0f}% Validation...")
    train_df, val_df = train_test_split(
        df_clean,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Paths for CSV saving
    full_csv = os.path.join(OUTPUT_DATA_DIR, 'neyshekar_cleaned.csv')
    train_csv = os.path.join(OUTPUT_DATA_DIR, 'train.csv')
    val_csv = os.path.join(OUTPUT_DATA_DIR, 'val.csv')
    
    print(f"[SAVING DATA] Saving CSV outputs to: {OUTPUT_DATA_DIR}...")
    df_clean.to_csv(full_csv, index=False, encoding='utf-8-sig')
    train_df.to_csv(train_csv, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_csv, index=False, encoding='utf-8-sig')
    
    # Optionally save a backup copy if NEYSHEKAR_BACKUP_DIR is set
    backup_dir = os.environ.get('NEYSHEKAR_BACKUP_DIR', None)
    if backup_dir:
        try:
            os.makedirs(backup_dir, exist_ok=True)
            df_clean.to_csv(os.path.join(backup_dir, 'neyshekar_cleaned.csv'), index=False, encoding='utf-8-sig')
            train_df.to_csv(os.path.join(backup_dir, 'train.csv'), index=False, encoding='utf-8-sig')
            val_df.to_csv(os.path.join(backup_dir, 'val.csv'), index=False, encoding='utf-8-sig')
            print(f"Backup copy saved to: {backup_dir}")
        except Exception as e:
            print(f"[WARNING] Could not save backup to '{backup_dir}': {e}")
    
    # Summary Report
    print("\n" + "="*70)
    print(" === NEYSHEKAR DATA PREPARATION & SPLIT SUMMARY ===")
    print("="*70)
    print(f"1. Total Raw Input Records:                {stats['initial_total']:,}")
    print(f"2. Corrupted Audio Files Dropped:           {stats['corrupted_count']:,}")
    print(f"3. Duration Outliers (< 1.0s) Dropped:      {stats['under_1s_count']:,}")
    print(f"4. Exact Row Duplicates Dropped:           {stats['exact_dups_dropped']:,}")
    print(f"5. Long Sentence Excess Copies Dropped:     {stats['long_excess_dropped']:,}")
    print("-"*70)
    print(f"TOTAL CLEANED RECORDS:                      {stats['final_clean_total']:,}")
    print(f"  --> Train Set (85%):                      {len(train_df):,} records")
    print(f"  --> Validation Set (15%):                 {len(val_df):,} records")
    print("="*70)
    print(f"Files saved successfully in project:\n  - {full_csv}\n  - {train_csv}\n  - {val_csv}")


if __name__ == '__main__':
    prepare_and_save_data()
