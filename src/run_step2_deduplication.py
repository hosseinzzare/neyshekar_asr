import sys
import os
import glob
import hashlib
import pandas as pd
import numpy as np

# Force UTF-8 encoding for standard output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in sys.path
sys.path.append(os.path.dirname(__file__))
from text_cleaner import (
    apply_step1_normalization,
    remove_diacritics,
    remove_spaces_and_zwnj
)

DATASET_PATH = r'E:\neyshekar dataset\data'
OUTPUT_DIR = r'E:\neyshekar dataset\investigation_results'
MAX_COPIES_PER_LONG_TEXT = 3

def compute_audio_hash(audio_dict):
    """Computes MD5 hash of audio bytes if available, else path."""
    if isinstance(audio_dict, dict):
        audio_bytes = audio_dict.get('bytes')
        if audio_bytes is not None:
            return hashlib.md5(audio_bytes).hexdigest()
        path = audio_dict.get('path')
        if path:
            return str(path)
    return str(audio_dict)

def run_deduplication_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*75)
    print(" === STEP 2: DUPLICATE DETECTION & MANAGEMENT PIPELINE (THRESHOLD=3) ===")
    print("="*75)
    
    # Load dataset
    parquet_files = sorted(glob.glob(os.path.join(DATASET_PATH, 'train-*.parquet')))
    print(f"\n[LOADING DATASET] Reading {len(parquet_files)} parquet files...")
    
    dfs = []
    for f in parquet_files:
        temp_df = pd.read_parquet(f, columns=['id', 'audio', 'text', 'duration'])
        dfs.append(temp_df)
    
    df = pd.concat(dfs, ignore_index=True)
    initial_total = len(df)
    print(f"Initial Total Dataset Records: {initial_total:,}")
    
    df['text_raw'] = df['text'].fillna('').astype(str)
    
    # -------------------------------------------------------------
    # STAGE 1: Exact Row Duplicates (Audio Hash + Text Raw)
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print(" STAGE 1: Checking Exact Row Duplicates (Audio Bytes/Path + Raw Text)")
    print("-"*75)
    
    df['audio_hash'] = df['audio'].apply(compute_audio_hash)
    exact_duplicates_mask = df.duplicated(subset=['audio_hash', 'text_raw'], keep='first')
    stage1_dropped_count = exact_duplicates_mask.sum()
    
    df_stage1 = df[~exact_duplicates_mask].copy().reset_index(drop=True)
    post_stage1_total = len(df_stage1)
    
    print(f"Exact Row Duplicates Found & Dropped:  {stage1_dropped_count:,}")
    print(f"Remaining Records after Stage 1:     {post_stage1_total:,}")
    
    # -------------------------------------------------------------
    # STAGE 2: Create Fingerprint Column
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print(" STAGE 2: Creating Text Fingerprint Column (Removing Spaces, ZWNJ, Diacritics, Punctuation)")
    print("-"*75)
    
    # Step 1 text (Arabic->Persian, Digits->Words, keeping spaces/ZWNJ for ASR training)
    df_stage1['text_step1'] = df_stage1['text_raw'].apply(apply_step1_normalization)
    
    # Fingerprint text (temporary column for duplicate matching only)
    df_stage1['text_fingerprint'] = df_stage1['text_step1'].apply(lambda x: remove_spaces_and_zwnj(remove_diacritics(x)))
    
    # -------------------------------------------------------------
    # STAGE 3: Threshold-Based Down-sampling (Max 3 Copies for Long Sentences)
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print(" STAGE 3: Threshold-Based Down-sampling (Max 3 Copies for Long Sentences)")
    print("-"*75)
    
    fp_counts = df_stage1['text_fingerprint'].value_counts()
    duplicate_fingerprints = set(fp_counts[fp_counts > 1].index)
    
    print(f"Total Unique Fingerprints:              {len(fp_counts):,}")
    print(f"Fingerprints appearing > 1 times:      {len(duplicate_fingerprints):,}")
    
    short_allowed_rows_count = 0
    short_unique_texts_count = 0
    
    long_small_duplicate_texts_count = 0  # Duplicates with <= 3 copies (all kept)
    long_small_duplicate_rows_count = 0
    
    long_heavy_duplicate_texts_count = 0  # Duplicates with > 3 copies (downsampled to 3)
    long_heavy_kept_rows_count = 0
    long_dropped_rows_count = 0
    
    rows_to_drop = []
    
    grouped = df_stage1.groupby('text_fingerprint')
    
    for fp, group_indices in grouped.groups.items():
        count = len(group_indices)
        if count > 1:
            first_idx = group_indices[0]
            first_text = df_stage1.loc[first_idx, 'text_step1']
            char_len = len(first_text)
            word_len = len(first_text.split())
            
            is_short = (char_len < 15) or (word_len < 4)
            
            if is_short:
                # Rule 1: Short text -> Keep ALL copies (Natural speech repetitions)
                short_allowed_rows_count += count
                short_unique_texts_count += 1
            else:
                # Rule 2: Long text Threshold Down-sampling (Cap = 3 copies)
                if count <= MAX_COPIES_PER_LONG_TEXT:
                    # 2 or 3 copies: Keep all copies for Acoustic Diversity!
                    long_small_duplicate_rows_count += count
                    long_small_duplicate_texts_count += 1
                else:
                    # > 3 copies: Keep top 3, drop 4th+ copies
                    drop_indices = group_indices[MAX_COPIES_PER_LONG_TEXT:]
                    rows_to_drop.extend(drop_indices)
                    
                    long_heavy_duplicate_texts_count += 1
                    long_heavy_kept_rows_count += MAX_COPIES_PER_LONG_TEXT
                    long_dropped_rows_count += len(drop_indices)

    df_final = df_stage1.drop(index=rows_to_drop).copy().reset_index(drop=True)
    final_total = len(df_final)
    
    print("\n" + "="*75)
    print(" === UPDATED DEDUPLICATION PIPELINE RESULTS (THRESHOLD = 3 COPIES) ===")
    print("="*75)
    print(f"1. Initial Raw Dataset Records:                     {initial_total:,}")
    print(f"2. Stage 1 - Exact Row Duplicates Dropped:           {stage1_dropped_count:,}")
    print(f"3. Short Duplicates Allowed & Kept (<15 chars):     {short_allowed_rows_count:,} rows ({short_unique_texts_count:,} unique texts)")
    print(f"4. Long Duplicates (2-3 copies) FULLY KEPT:         {long_small_duplicate_rows_count:,} rows ({long_small_duplicate_texts_count:,} unique texts)")
    print(f"5. Long Duplicates (>3 copies) DOWNSAMPLED TO 3:    {long_heavy_kept_rows_count:,} rows kept ({long_heavy_duplicate_texts_count:,} unique texts)")
    print(f"   --> DROPPED Excess Copies (4th, 5th, etc.):       {long_dropped_rows_count:,} rows dropped")
    print("-"*75)
    print(f"FINAL CLEANED DATASET SIZE:                          {final_total:,} rows")
    print(f"Total Net Reduction:                                {initial_total - final_total:,} rows ({(initial_total - final_total)/initial_total*100:.2f}%)")
    print("="*75)

    # -------------------------------------------------------------
    # Save Final Cleaned Dataset to Drive E:
    # -------------------------------------------------------------
    output_parquet = os.path.join(OUTPUT_DIR, 'dataset_deduplicated_final.parquet')
    df_save = df_final[['id', 'audio', 'text_raw', 'text_step1', 'duration']]
    df_save.to_parquet(output_parquet, index=False)
    print(f"\n[SAVED] Threshold-downsampled dataset successfully saved to Drive E:\n  --> {output_parquet}")

    # Save summary report to Drive E:
    report_file = os.path.join(OUTPUT_DIR, 'deduplication_pipeline_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Neyshekar Dataset - Threshold Down-sampling (Max 3 Copies) Summary Report\n")
        f.write("=======================================================================\n\n")
        f.write(f"Initial Total Records:                     {initial_total}\n")
        f.write(f"Stage 1 Exact Row Duplicates Dropped:      {stage1_dropped_count}\n")
        f.write(f"Short Duplicates Kept (<15 chars):         {short_allowed_rows_count} (from {short_unique_texts_count} unique texts)\n")
        f.write(f"Long Duplicates (2-3 copies) Fully Kept:   {long_small_duplicate_rows_count} (from {long_small_duplicate_texts_count} unique texts)\n")
        f.write(f"Long Duplicates (>3 copies) Kept (3 per text): {long_heavy_kept_rows_count} (from {long_heavy_duplicate_texts_count} unique texts)\n")
        f.write(f"Long Duplicate Excess Rows Dropped:        {long_dropped_rows_count}\n")
        f.write(f"Final Remaining Clean Records:             {final_total}\n")
        f.write(f"Saved Parquet Path:                        {output_parquet}\n")

    print(f"Report saved to Drive E: {report_file}")

if __name__ == '__main__':
    run_deduplication_pipeline()
