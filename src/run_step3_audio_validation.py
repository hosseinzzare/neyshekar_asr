import sys
import os
import io
import hashlib
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns

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

# The dataset location is supplied at run time rather than written into the source; see
# paths.py for the resolution order.
from paths import resolve_paths, find_shards
DATASET_PATH, OUTPUT_DIR = resolve_paths()


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

def run_audio_validation_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*75)
    print(" === TASK 1: AUDIO VALIDATION & DURATION DISTRIBUTION ANALYSIS ===")
    print("="*75)
    
    # Load raw dataset
    parquet_files = find_shards(DATASET_PATH)
    print(f"\n[1/4] Loading {len(parquet_files)} raw dataset parquet files...")
    
    dfs = []
    for f in parquet_files:
        temp_df = pd.read_parquet(f, columns=['id', 'audio', 'text', 'duration'])
        dfs.append(temp_df)
    
    df = pd.concat(dfs, ignore_index=True)
    initial_total = len(df)
    print(f"Total Initial Dataset Records: {initial_total:,}")
    
    df['text_raw'] = df['text'].fillna('').astype(str)
    
    # -------------------------------------------------------------
    # STAGE 1: Audio Health Check & Duration Calculation via soundfile
    # -------------------------------------------------------------
    print("\n[2/4] Performing Audio Integrity & Fast Header Duration Extraction...")
    
    corrupted_indices = []
    audio_durations = []
    
    for idx, row in df.iterrows():
        audio_obj = row['audio']
        try:
            if not isinstance(audio_obj, dict) or 'bytes' not in audio_obj or not audio_obj['bytes']:
                corrupted_indices.append(idx)
                audio_durations.append(np.nan)
                continue
            
            # Read header info efficiently using soundfile
            info = sf.info(io.BytesIO(audio_obj['bytes']))
            audio_durations.append(info.duration)
        except Exception:
            corrupted_indices.append(idx)
            audio_durations.append(np.nan)
            
    df['audio_duration'] = audio_durations
    corrupted_count = len(corrupted_indices)
    
    # Identify duration outliers
    under_1s_mask = df['audio_duration'] < 1.0
    over_30s_mask = df['audio_duration'] > 30.0
    
    under_1s_count = under_1s_mask.sum()
    over_30s_count = over_30s_mask.sum()
    valid_audio_mask = (df['audio_duration'] >= 1.0) & (df['audio_duration'] <= 30.0) & (~df['audio_duration'].isna())
    
    print("\n" + "-"*75)
    print(" AUDIO INTEGRITY AUDIT RESULTS:")
    print("-"*75)
    print(f"1. Corrupted / Unreadable Audio Files:    {corrupted_count:,}")
    print(f"2. Audio Duration Under 1.0s (< 1s):      {under_1s_count:,}")
    print(f"3. Audio Duration Over 30.0s (> 30s):     {over_30s_count:,}")
    print(f"4. Healthy & Valid Duration Range (1s-30s):{valid_audio_mask.sum():,}")
    print("-"*75)
    
    # -------------------------------------------------------------
    # STAGE 2: Deduplication (Stage 1 + Stage 2 + Stage 3 Downsampling)
    # -------------------------------------------------------------
    print("\n[3/4] Applying Text Normalization & Threshold-Based Deduplication...")
    
    # Filter out corrupted & invalid duration audio first
    df_valid_audio = df[valid_audio_mask].copy().reset_index(drop=True)
    
    # Exact Row Duplicates (Audio Hash + Text Raw)
    df_valid_audio['audio_hash'] = df_valid_audio['audio'].apply(compute_audio_hash)
    exact_duplicates_mask = df_valid_audio.duplicated(subset=['audio_hash', 'text_raw'], keep='first')
    exact_row_duplicates_dropped = exact_duplicates_mask.sum()
    
    df_dedup_stage1 = df_valid_audio[~exact_duplicates_mask].copy().reset_index(drop=True)
    
    # Text Normalization & Fingerprint Creation
    df_dedup_stage1['text_step1'] = df_dedup_stage1['text_raw'].apply(apply_step1_normalization)
    df_dedup_stage1['text_fingerprint'] = df_dedup_stage1['text_step1'].apply(lambda x: remove_spaces_and_zwnj(remove_diacritics(x)))
    
    # Deduplication Grouping
    rows_to_drop = []
    short_kept_count = 0
    long_kept_count = 0
    long_dropped_count = 0
    
    grouped = df_dedup_stage1.groupby('text_fingerprint')
    for fp, group_indices in grouped.groups.items():
        count = len(group_indices)
        if count > 1:
            first_text = df_dedup_stage1.loc[group_indices[0], 'text_step1']
            is_short = (len(first_text) < 15) or (len(first_text.split()) < 4)
            if is_short:
                short_kept_count += count
            else:
                if count <= MAX_COPIES_PER_LONG_TEXT:
                    long_kept_count += count
                else:
                    drop_indices = group_indices[MAX_COPIES_PER_LONG_TEXT:]
                    rows_to_drop.extend(drop_indices)
                    long_kept_count += MAX_COPIES_PER_LONG_TEXT
                    long_dropped_count += len(drop_indices)

    df_final = df_dedup_stage1.drop(index=rows_to_drop).copy().reset_index(drop=True)
    final_clean_total = len(df_final)
    
    # -------------------------------------------------------------
    # STAGE 3: Visualization (Audio Duration Histogram)
    # -------------------------------------------------------------
    print("\n[4/4] Plotting Audio Duration Histogram...")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Histogram plot
    ax = sns.histplot(
        df_valid_audio['audio_duration'],
        bins=60,
        kde=True,
        color='#2b5c8f',
        edgecolor='black',
        alpha=0.75
    )
    
    # Draw Vertical dashed lines at 1.0s and 30.0s
    plt.axvline(x=1.0, color='#e74c3c', linestyle='--', linewidth=2.5, label='Whisper Valid Range (1s - 30s)')
    plt.axvline(x=30.0, color='#e74c3c', linestyle='--', linewidth=2.5)
    
    plt.title('Neyshekar Dataset — Audio Duration Distribution', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Audio Duration (Seconds)', fontsize=12, labelpad=10)
    plt.ylabel('Number of Audio Samples', fontsize=12, labelpad=10)
    plt.xlim(0, 32)
    
    # Annotate stats on chart
    mean_dur = df_valid_audio['audio_duration'].mean()
    median_dur = df_valid_audio['audio_duration'].median()
    plt.axvline(x=mean_dur, color='#27ae60', linestyle='-', linewidth=1.8, label=f'Mean Duration ({mean_dur:.2f}s)')
    
    plt.legend(loc='upper right', fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, 'audio_duration_histogram.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Histogram saved successfully to\n  --> {plot_path}")
    
    # -------------------------------------------------------------
    # Summary Report & Output Log
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print(" === FINAL AUDIO VALIDATION & DEDUPLICATION SUMMARY ===")
    print("="*75)
    print(f"1. Total Initial Records:                {initial_total:,}")
    print(f"2. Corrupted Audio Files:                 {corrupted_count:,}")
    print(f"3. Audio Files Under 1.0s (< 1s):         {under_1s_count:,}")
    print(f"4. Audio Files Over 30.0s (> 30s):        {over_30s_count:,}")
    print(f"5. Exact Row Duplicates Dropped:         {exact_row_duplicates_dropped:,}")
    print(f"6. Long Sentence Excess Copies Dropped:   {long_dropped_count:,}")
    print("-"*75)
    print(f"FINAL CLEAN & VALID RECORDS REMAINING:    {final_clean_total:,}")
    print("="*75)
    
    # Save Report to the output directory
    report_path = os.path.join(OUTPUT_DIR, 'audio_validation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Neyshekar Dataset - Audio Validation & Duration Distribution Report\n")
        f.write("===================================================================\n\n")
        f.write(f"Total Initial Records:                {initial_total}\n")
        f.write(f"Corrupted Audio Files:                {corrupted_count}\n")
        f.write(f"Audio Files Under 1.0s (< 1s):        {under_1s_count}\n")
        f.write(f"Audio Files Over 30.0s (> 30s):       {over_30s_count}\n")
        f.write(f"Exact Row Duplicates Dropped:         {exact_row_duplicates_dropped}\n")
        f.write(f"Long Sentence Excess Copies Dropped:  {long_dropped_count}\n")
        f.write(f"Final Clean & Valid Records Remaining:{final_clean_total}\n")
        f.write(f"Mean Audio Duration:                  {mean_dur:.2f} seconds\n")
        f.write(f"Median Audio Duration:                {median_dur:.2f} seconds\n")
        f.write(f"Histogram Image Saved At:             {plot_path}\n")

    print(f"Report saved to {report_path}")

if __name__ == '__main__':
    run_audio_validation_pipeline()
