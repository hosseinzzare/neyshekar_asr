"""
Extract a Test Clip for the Inference Service
=============================================

The Neyshekar dataset ships as parquet shards with the audio stored inline as bytes; there are
no loose .wav files anywhere in it. This pulls one clip out and writes it to disk so the
container can be tested against real data rather than an arbitrary recording.

The clip is taken from the VALIDATION split, so it is audio the adapter was never trained on.
The reference transcript is printed alongside, which is the point: a transcription is only
evidence that the service works if there is something to compare it against.

Usage
-----
    python make_test_clip.py --dataset_path "E:/neyshekar dataset"
    python make_test_clip.py --dataset_path "E:/neyshekar dataset" --index 7
    python make_test_clip.py --dataset_path "E:/neyshekar dataset" --n 5   # write five clips

Then:
    curl.exe -F "file=@test_clips/val_37678.wav" http://localhost:8000/transcribe
"""

import argparse
import io
import os
import sys

import pandas as pd


def find_parquet(dataset_path):
    """
    Defer to the same resolver the Task 1 scripts use, so this script and those cannot disagree
    about where the corpus is. An earlier version swept recursively for *.parquet, which found
    the shards but would also have picked up validation shards and any intermediate parquet a
    previous step had written.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from paths import find_shards
    return find_shards(dataset_path)


def audio_column(df):
    """
    Find the column holding the audio. It is a struct of {bytes, path} rather than a plain
    array, so a dtype check will not find it -- inspect one value instead.
    """
    for col in df.columns:
        v = df[col].iloc[0]
        if isinstance(v, dict) and "bytes" in v:
            return col
    sys.exit(f"No audio column found. Columns present: {list(df.columns)}")


def main():
    p = argparse.ArgumentParser(description="Write validation clips out as .wav for testing")
    p.add_argument("--dataset_path", default=os.environ.get("NEYSHEKAR_RAW_DIR"),
                   help="directory holding train-*.parquet (or set NEYSHEKAR_RAW_DIR)")
    p.add_argument("--val_csv", default="data/val.csv")
    p.add_argument("--out_dir", default="test_clips")
    p.add_argument("--index", type=int, default=0, help="which validation row to start from")
    p.add_argument("--n", type=int, default=1, help="how many clips to write")
    args = p.parse_args()

    if not args.dataset_path:
        sys.exit("Give --dataset_path, e.g.  --dataset_path \"E:/neyshekar dataset\"")

    val = pd.read_csv(args.val_csv)
    wanted = val.iloc[args.index:args.index + args.n]
    if wanted.empty:
        sys.exit(f"--index {args.index} is past the end of {args.val_csv} ({len(val):,} rows)")
    targets = dict(zip(wanted["id"], wanted["cleaned_text"]))
    print(f"Looking for {len(targets)} validation clip(s): ids {list(targets)}")

    os.makedirs(args.out_dir, exist_ok=True)
    files = find_parquet(args.dataset_path)
    print(f"Scanning {len(files)} parquet shard(s)...")

    found = {}
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        hits = df[df["id"].isin(targets)]
        if not hits.empty:
            col = audio_column(df)
            for _, row in hits.iterrows():
                found[row["id"]] = row[col]
        print(f"  shard {i}/{len(files)}  ({len(found)}/{len(targets)} found)", end="\r")
        if len(found) == len(targets):
            break
    print()

    if not found:
        sys.exit("None of those ids are in the parquet shards -- wrong dataset version?")

    # soundfile is imported here rather than at the top so the script still reports a useful
    # error about paths on a machine where it is not installed.
    import soundfile as sf

    print()
    for sample_id, blob in found.items():
        audio, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32")
        out = os.path.join(args.out_dir, f"val_{sample_id}.wav")
        sf.write(out, audio, sr)
        print(f"  wrote {out}   {len(audio) / sr:.2f}s @ {sr} Hz")
        print(f"  reference: {targets[sample_id]}")
        print()

    first = os.path.join(args.out_dir, f"val_{list(found)[0]}.wav").replace("\\", "/")
    print("Test the running container with:")
    print(f'  curl.exe -F "file=@{first}" http://localhost:8000/transcribe')


if __name__ == "__main__":
    main()
