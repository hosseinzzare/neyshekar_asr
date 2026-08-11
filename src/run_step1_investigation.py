import sys
import os
import pandas as pd
import numpy as np

# Force UTF-8 encoding for standard output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in sys.path
sys.path.append(os.path.dirname(__file__))
from text_cleaner import (
    convert_arabic_to_persian,
    lexicalize_numbers,
    apply_step1_normalization,
    has_arabic_chars,
    has_digits
)

# The dataset location is supplied at run time rather than written into the source; see
# paths.py for the resolution order.
from paths import resolve_paths, find_shards
DATASET_PATH, OUTPUT_DIR = resolve_paths()


def run_investigation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading dataset parquet files from: {DATASET_PATH}...")
    
    parquet_files = find_shards(DATASET_PATH)
    if not parquet_files:
        print(f"Error: No parquet files found in {DATASET_PATH}")
        return
    
    print(f"Found {len(parquet_files)} parquet files. Reading columns...")
    dfs = [pd.read_parquet(f, columns=['id', 'text', 'duration']) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    total_records = len(df)
    print(f"Total dataset records: {total_records:,}")
    
    df['text_raw'] = df['text'].fillna('').astype(str)
    
    # 1. Audit Arabic Characters & Digits
    df['has_arabic'] = df['text_raw'].apply(has_arabic_chars)
    df['has_digits'] = df['text_raw'].apply(has_digits)
    
    arabic_count = df['has_arabic'].sum()
    arabic_pct = (arabic_count / total_records) * 100
    digits_count = df['has_digits'].sum()
    digits_pct = (digits_count / total_records) * 100
    raw_duplicates_count = df['text_raw'].duplicated().sum()
    raw_duplicates_pct = (raw_duplicates_count / total_records) * 100
    
    print("\n" + "="*65)
    print(" === STEP 1: RAW DATASET TEXT AUDIT ===")
    print("="*65)
    print(f"Total Audio Samples:                    {total_records:,}")
    print(f"Transcripts with Arabic Encodings (ي,ك):  {arabic_count:,} ({arabic_pct:.2f}%)")
    print(f"Transcripts with Numeric Digits (0-9):   {digits_count:,} ({digits_pct:.2f}%)")
    print(f"Raw Exact Duplicate Transcripts:        {raw_duplicates_count:,} ({raw_duplicates_pct:.2f}%)")
    print("="*65)
    
    # Apply Step 1 Normalization
    print("\nApplying Step 1 Normalization (Arabic->Persian & Digits->Persian Words)...")
    df['text_step1'] = df['text_raw'].apply(apply_step1_normalization)
    
    # Verify post-normalization counts
    remaining_arabic = df['text_step1'].apply(has_arabic_chars).sum()
    remaining_digits = df['text_step1'].apply(has_digits).sum()
    step1_duplicates_count = df['text_step1'].duplicated().sum()
    step1_duplicates_pct = (step1_duplicates_count / total_records) * 100
    
    print("\n" + "="*65)
    print(" === POST STEP 1 NORMALIZATION VERIFICATION ===")
    print("="*65)
    print(f"Remaining Arabic Characters (Target: 0):  {remaining_arabic:,}")
    print(f"Remaining Numeric Digits (Target: 0):     {remaining_digits:,}")
    print(f"Exact Duplicate Transcripts Post Step 1: {step1_duplicates_count:,} ({step1_duplicates_pct:.2f}%)")
    print("="*65)

    # SAVE THE STEP 1 INTERMEDIATE PARQUET
    output_parquet = os.path.join(OUTPUT_DIR, 'dataset_step1_normalized.parquet')
    df_to_save = df[['id', 'duration', 'text_raw', 'text_step1']]
    df_to_save.to_parquet(output_parquet, index=False)
    print(f"\n[SAVED] Step 1 normalized dataset successfully saved to\n  --> {output_parquet}")

    # Save summary report to the output directory
    report_file = os.path.join(OUTPUT_DIR, 'step1_investigation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Neyshekar Dataset - Step 1 Text Investigation Report\n")
        f.write("====================================================\n\n")
        f.write(f"Total Records: {total_records}\n")
        f.write(f"Raw Arabic Encodings: {arabic_count} ({arabic_pct:.2f}%)\n")
        f.write(f"Raw Digits: {digits_count} ({digits_pct:.2f}%)\n")
        f.write(f"Raw Duplicates: {raw_duplicates_count} ({raw_duplicates_pct:.2f}%)\n")
        f.write(f"Remaining Arabic: {remaining_arabic}\n")
        f.write(f"Remaining Digits: {remaining_digits}\n")
        f.write(f"Post Step 1 Duplicates: {step1_duplicates_count} ({step1_duplicates_pct:.2f}%)\n")
        f.write(f"Saved Processed File: {output_parquet}\n")
    
    print(f"Report successfully saved to {report_file}")

if __name__ == '__main__':
    run_investigation()
