import sys
import os
import io
import glob
import pandas as pd
import numpy as np
import soundfile as sf

# Force UTF-8 encoding for standard output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Supplied at run time; see paths.py.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths import resolve_paths
DATASET_PATH, OUTPUT_DIR = resolve_paths()


def test_audio_health():
    print("Loading raw parquet files to test audio integrity and duration...")
    parquet_files = sorted(glob.glob(os.path.join(DATASET_PATH, 'train-*.parquet')))
    
    corrupted_count = 0
    under_1s_count = 0
    over_30s_count = 0
    valid_count = 0
    
    durations = []
    
    for f_idx, f_path in enumerate(parquet_files):
        print(f"Processing file {f_idx+1}/{len(parquet_files)}: {os.path.basename(f_path)}...")
        df = pd.read_parquet(f_path, columns=['id', 'audio', 'text', 'duration'])
        
        for idx, row in df.iterrows():
            audio_obj = row['audio']
            
            # Step 1: Health Check
            try:
                if not isinstance(audio_obj, dict) or 'bytes' not in audio_obj or not audio_obj['bytes']:
                    corrupted_count += 1
                    continue
                
                audio_bytes = audio_obj['bytes']
                # Fast header read without decoding PCM array
                info = sf.info(io.BytesIO(audio_bytes))
                dur = info.duration
                durations.append(dur)
                
                # Step 3: Outlier Check
                if dur < 1.0:
                    under_1s_count += 1
                elif dur > 30.0:
                    over_30s_count += 1
                else:
                    valid_count += 1
                    
            except Exception as e:
                corrupted_count += 1

    print("\n" + "="*65)
    print(" === AUDIO INTEGRITY & DURATION AUDIT RESULTS ===")
    print("="*65)
    print(f"Total Samples Inspected:        {len(durations) + corrupted_count:,}")
    print(f"Corrupted Audio Files:           {corrupted_count:,}")
    print(f"Under 1.0s Duration (< 1s):      {under_1s_count:,}")
    print(f"Over 30.0s Duration (> 30s):     {over_30s_count:,}")
    print(f"Healthy & Valid Range (1s-30s):  {valid_count:,}")
    print("="*65)

if __name__ == '__main__':
    test_audio_health()
