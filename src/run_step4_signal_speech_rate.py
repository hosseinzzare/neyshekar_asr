import sys
import os
import io
import hashlib
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

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
    if isinstance(audio_dict, dict):
        audio_bytes = audio_dict.get('bytes')
        if audio_bytes is not None:
            return hashlib.md5(audio_bytes).hexdigest()
        path = audio_dict.get('path')
        if path:
            return str(path)
    return str(audio_dict)

def analyze_signal_batch(audio_bytes_list):
    results = []
    for b in audio_bytes_list:
        try:
            data, sr = sf.read(io.BytesIO(b), dtype='float32')
            if len(data) == 0:
                results.append((True, False, 0.0, 0.0))
                continue
            abs_data = np.abs(data)
            max_amp = float(np.max(abs_data))
            rms = float(np.sqrt(np.mean(data**2)))
            
            is_silent = (max_amp < 0.001) or (rms < 0.0001)
            is_clipped = (max_amp >= 0.99)
            results.append((is_silent, is_clipped, max_amp, rms))
        except Exception:
            results.append((True, False, 0.0, 0.0))
    return results

def run_speech_rate_and_signal_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*75)
    print(" === TASK 1: SPEECH RATE (WPS/CPS) & SIGNAL INTEGRITY ANALYSIS ===")
    print("="*75)
    
    # -------------------------------------------------------------
    # 1. Load Parquet & Run Deduplication + Audio Validation Pipeline
    # -------------------------------------------------------------
    parquet_files = find_shards(DATASET_PATH)
    print(f"\n[1/4] Loading {len(parquet_files)} dataset parquet files...")
    
    dfs = [pd.read_parquet(f, columns=['id', 'audio', 'text', 'duration']) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    initial_total = len(df)
    
    df['text_raw'] = df['text'].fillna('').astype(str)
    
    # Fast header duration extraction
    audio_durations = []
    for idx, row in df.iterrows():
        try:
            info = sf.info(io.BytesIO(row['audio']['bytes']))
            audio_durations.append(info.duration)
        except Exception:
            audio_durations.append(np.nan)
    df['audio_duration'] = audio_durations
    
    # Filter valid audio duration (1.0s <= duration <= 30.0s)
    valid_audio_mask = (df['audio_duration'] >= 1.0) & (df['audio_duration'] <= 30.0) & (~df['audio_duration'].isna())
    df_valid = df[valid_audio_mask].copy().reset_index(drop=True)
    
    # Deduplication
    df_valid['audio_hash'] = df_valid['audio'].apply(compute_audio_hash)
    df_valid = df_valid[~df_valid.duplicated(subset=['audio_hash', 'text_raw'], keep='first')].reset_index(drop=True)
    
    df_valid['text_step1'] = df_valid['text_raw'].apply(apply_step1_normalization)
    df_valid['text_fingerprint'] = df_valid['text_step1'].apply(lambda x: remove_spaces_and_zwnj(remove_diacritics(x)))
    
    rows_to_drop = []
    grouped = df_valid.groupby('text_fingerprint')
    for fp, group_indices in grouped.groups.items():
        count = len(group_indices)
        if count > 1:
            first_text = df_valid.loc[group_indices[0], 'text_step1']
            is_short = (len(first_text) < 15) or (len(first_text.split()) < 4)
            if not is_short and count > MAX_COPIES_PER_LONG_TEXT:
                rows_to_drop.extend(group_indices[MAX_COPIES_PER_LONG_TEXT:])
                
    df_clean = df_valid.drop(index=rows_to_drop).copy().reset_index(drop=True)
    total_clean = len(df_clean)
    print(f"Dataset Records to Analyze: {total_clean:,}")

    # -------------------------------------------------------------
    # STAGE 1: Speech Rate Anomaly Detection (WPS / CPS) & Plotting
    # -------------------------------------------------------------
    print("\n[2/4] Calculating Word Count, Char Count, WPS, and CPS metrics...")
    
    df_clean['word_count'] = df_clean['text_step1'].str.split().str.len()
    df_clean['char_count'] = df_clean['text_step1'].str.len()
    
    df_clean['wps'] = df_clean['word_count'] / df_clean['audio_duration']
    df_clean['cps'] = df_clean['char_count'] / df_clean['audio_duration']
    
    # Suspicious thresholds: WPS > 5.0 (too fast / audio cut off), CPS < 8.0 (too slow / long silence)
    high_wps_mask = df_clean['wps'] > 5.0
    low_cps_mask = df_clean['cps'] < 8.0
    suspicious_speech_rate_mask = high_wps_mask | low_cps_mask
    
    high_wps_count = high_wps_mask.sum()
    low_cps_count = low_cps_mask.sum()
    suspicious_speech_rate_count = suspicious_speech_rate_mask.sum()
    
    print("\n" + "-"*75)
    print(" SPEECH RATE (WPS / CPS) ANOMALY RESULTS:")
    print("-"*75)
    print(f"1. High WPS Outliers (WPS > 5.0 words/s):   {high_wps_count:,} ({high_wps_count/total_clean*100:.2f}%)")
    print(f"2. Low CPS Outliers (CPS < 8.0 chars/s):    {low_cps_count:,} ({low_cps_count/total_clean*100:.2f}%)")
    print(f"3. Total Suspicious Speech Rate Samples:    {suspicious_speech_rate_count:,} ({suspicious_speech_rate_count/total_clean*100:.2f}%)")
    print("-"*75)

    # Plot Transcript Length Histogram (Word Count)
    print("Plotting Transcript Length Histogram...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    ax = sns.histplot(
        df_clean['word_count'],
        bins=50,
        kde=True,
        color='#e67e22',
        edgecolor='black',
        alpha=0.75
    )
    
    mean_words = df_clean['word_count'].mean()
    median_words = df_clean['word_count'].median()
    
    plt.axvline(x=mean_words, color='#c0392b', linestyle='--', linewidth=2.0, label=f'Mean Length ({mean_words:.2f} words)')
    plt.axvline(x=median_words, color='#27ae60', linestyle='-', linewidth=2.0, label=f'Median Length ({median_words:.0f} words)')
    
    plt.title('Neyshekar Dataset — Transcript Word Count Distribution', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Transcript Length (Word Count)', fontsize=12, labelpad=10)
    plt.ylabel('Number of Samples', fontsize=12, labelpad=10)
    plt.legend(loc='upper right', fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    length_plot_path = os.path.join(OUTPUT_DIR, 'transcript_length_histogram.png')
    plt.savefig(length_plot_path, dpi=300)
    plt.close()
    print(f"Transcript length plot saved at:\n  --> {length_plot_path}")

    # -------------------------------------------------------------
    # STAGE 2: Parallel Signal Processing (Silence & Clipping Detection)
    # -------------------------------------------------------------
    print("\n[3/4] Running Parallel Signal Analysis (Absolute Silence & Clipping Detection)...")
    
    audio_bytes_list = [row['audio']['bytes'] for idx, row in df_clean.iterrows()]
    
    # Split audio bytes into chunks for multi-core processing
    num_cores = os.cpu_count() or 4
    chunk_size = int(np.ceil(len(audio_bytes_list) / num_cores))
    chunks = [audio_bytes_list[i:i+chunk_size] for i in range(0, len(audio_bytes_list), chunk_size)]
    
    print(f"Processing {len(audio_bytes_list):,} audio signals using {num_cores} CPU worker threads...")
    parallel_results = Parallel(n_jobs=num_cores)(delayed(analyze_signal_batch)(chunk) for chunk in chunks)
    
    # Flatten results
    flattened_results = [item for sublist in parallel_results for item in sublist]
    
    is_silent_list = [r[0] for r in flattened_results]
    is_clipped_list = [r[1] for r in flattened_results]
    max_amp_list = [r[2] for r in flattened_results]
    rms_list = [r[3] for r in flattened_results]
    
    df_clean['is_silent'] = is_silent_list
    df_clean['is_clipped'] = is_clipped_list
    df_clean['max_amplitude'] = max_amp_list
    df_clean['rms_energy'] = rms_list
    
    silent_count = df_clean['is_silent'].sum()
    clipped_count = df_clean['is_clipped'].sum()
    
    print("\n" + "-"*75)
    print(" SIGNAL QUALITY AUDIT RESULTS:")
    print("-"*75)
    print(f"1. Absolute Silence Audio Files (< 0.001 Amp / RMS < 0.0001): {silent_count:,}")
    print(f"2. Clipped / Distorted Audio Files (Max Amp >= 0.99):        {clipped_count:,}")
    print("-"*75)

    # -------------------------------------------------------------
    # STAGE 3: Final Report & Data Export
    # -------------------------------------------------------------
    print("\n[4/4] Exporting Summary Report to the output directory...")
    
    report_file = os.path.join(OUTPUT_DIR, 'speech_rate_signal_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Neyshekar Dataset - Speech Rate & Signal Quality Investigation Report\n")
        f.write("===================================================================\n\n")
        f.write(f"Total Clean Records Analyzed:              {total_clean}\n")
        f.write(f"High WPS Outliers (WPS > 5.0):             {high_wps_count} ({high_wps_count/total_clean*100:.2f}%)\n")
        f.write(f"Low CPS Outliers (CPS < 8.0):              {low_cps_count} ({low_cps_count/total_clean*100:.2f}%)\n")
        f.write(f"Total Suspicious Speech Rate Samples:      {suspicious_speech_rate_count} ({suspicious_speech_rate_count/total_clean*100:.2f}%)\n")
        f.write(f"Absolute Silence Audio Files:               {silent_count}\n")
        f.write(f"Clipped / Distorted Audio Files:           {clipped_count}\n")
        f.write(f"Transcript Mean Word Length:               {mean_words:.2f} words\n")
        f.write(f"Transcript Median Word Length:             {median_words:.0f} words\n")
        f.write(f"Saved Plot Path:                           {length_plot_path}\n")

    print("\n" + "="*75)
    print(" === FINAL INVESTIGATION SUMMARY ===")
    print("="*75)
    print(f"1. Total Records Analyzed:              {total_clean:,}")
    print(f"2. Suspicious Speech Rate (WPS/CPS):    {suspicious_speech_rate_count:,}")
    print(f"3. Absolute Silence Files:              {silent_count:,}")
    print(f"4. Clipped / Distorted Audio Files:     {clipped_count:,}")
    print(f"Report saved to: {report_file}")
    print("="*75)

if __name__ == '__main__':
    run_speech_rate_and_signal_analysis()
