"""
Task 1 Verification Script
==========================

Re-derives every data-cleaning statistic claimed in the Task 1 report directly from the
RAW parquet files, then compares them against the report's numbers and against the
cleaned CSVs that Task 2 actually trains on.

The point is reproducibility: an assessor should be able to run this and get the same
figures that appear in the written report. Any line marked MISMATCH needs the report
corrected before submission.

Deliberately imports the cleaning helpers from src/ rather than reimplementing them,
so this verifies the ACTUAL pipeline logic rather than a lookalike.

Usage
-----
    # full check (reads audio bytes to detect exact duplicates -- slower, ~7 GB read)
    python verify_task1.py --raw_dir "E:\\neyshekar dataset"

    # fast check (skips audio hashing; everything except the exact-duplicate count)
    python verify_task1.py --raw_dir "E:\\neyshekar dataset" --skip_audio
"""

import argparse
import glob
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from text_cleaner import (  # noqa: E402
    apply_step1_normalization,
    remove_diacritics,
    remove_spaces_and_zwnj,
)

# text_cleaner degrades SILENTLY when num2fawords is absent: Arabic->Persian still runs,
# but digit lexicalization ("۱۲" -> "دوازده") is skipped with no warning. Replaying the
# pipeline in that state would diverge from the original data_prep.py run and produce
# bogus MISMATCH lines below. Refuse to run rather than report misleading numbers.
try:
    import num2fawords  # noqa: F401
except ImportError:
    sys.exit(
        "ERROR: num2fawords is not installed.\n"
        "  Without it, text normalization silently skips digit-to-word conversion, so this\n"
        "  script would NOT reproduce the original cleaning run and every comparison below\n"
        "  would be untrustworthy.\n"
        "  Fix:  pip install num2fawords"
    )

# ----------------------------------------------------------------------------------
# Figures claimed in the Task 1 report, so the script can flag its own disagreements.
# ----------------------------------------------------------------------------------
REPORT = {
    "raw_total": 40008,
    "arabic_rows": 1480,
    "digit_rows": 1191,
    "corrupted": 0,
    "over_30s": 0,
    "under_1s": 10,
    "exact_dups": 134,
    "long_excess_dropped": 532,
    "final_total": 39332,
    "max_duration": 24.36,
    "mean_duration": 5.67,
    "low_cps": 10983,
    "clipped": 8693,
}

ARABIC = re.compile(r"[\u064a\u0643\u0629\u0623\u0625\u0671\u0624\u0626\u0649]")
DIGITS = re.compile(r"[\d\u06f0-\u06f9\u0660-\u0669]")

MAX_COPIES_PER_LONG_TEXT = 3


def fingerprint(text: str) -> str:
    """Identical to data_prep.py: strip diacritics, punctuation, spaces and ZWNJ."""
    return remove_spaces_and_zwnj(remove_diacritics(text))


def is_short(text: str) -> bool:
    """data_prep.py's exemption rule for high-frequency short conversational phrases."""
    return (len(text) < 15) or (len(text.split()) < 4)


def load_raw(raw_dir: str, skip_audio: bool) -> pd.DataFrame:
    # Look in the given folder first, then one level down (HF layouts keep shards in data/),
    # then anywhere beneath it. 'investigation_results/' holds Task 1 intermediates, not raw
    # shards, so it is excluded to avoid mixing pre-cleaned data into the raw baseline.
    files = sorted(glob.glob(os.path.join(raw_dir, "train-*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(raw_dir, "data", "train-*.parquet")))
    if not files:
        files = sorted(
            f for f in glob.glob(os.path.join(raw_dir, "**", "train-*.parquet"), recursive=True)
            if "investigation_results" not in f.replace("\\", "/")
        )
    if not files:
        raise FileNotFoundError(
            f"No train-*.parquet files found under {raw_dir!r} (searched recursively).\n"
            f"Point --raw_dir at the dataset root or the folder holding the 15 shards."
        )
    print(f"[LOAD] Found {len(files)} parquet shard(s) under {raw_dir}")
    print(f"       first: {files[0]}")

    cols = ["id", "text", "duration"] + ([] if skip_audio else ["audio"])
    frames = []
    for i, path in enumerate(files, 1):
        df = pd.read_parquet(path, columns=cols)
        if not skip_audio:
            # Hash immediately, then drop the bytes so memory stays flat.
            df["audio_hash"] = df["audio"].apply(
                lambda a: hashlib.md5(a["bytes"]).hexdigest()
                if isinstance(a, dict) and a.get("bytes") is not None
                else str(a)
            )
            df = df.drop(columns=["audio"])
        frames.append(df)
        print(f"       shard {i}/{len(files)}: {len(df):,} rows")
    out = pd.concat(frames, ignore_index=True)
    out["text_raw"] = out["text"].fillna("").astype(str)
    return out


def row(label, actual, expected, fmt="{:,}", tol=0):
    """Print one comparison line, flagging disagreement with the report."""
    a = fmt.format(actual)
    if expected is None:
        print(f"  {label:<44} {a:>12}")
        return True
    e = fmt.format(expected)
    ok = abs(actual - expected) <= tol
    print(f"  {label:<44} {a:>12}   report: {e:>10}  {'OK' if ok else '<-- MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Verify Task 1 cleaning statistics")
    ap.add_argument("--raw_dir", default=os.environ.get("NEYSHEKAR_RAW_DIR"),
                    help="Folder containing the raw train-*.parquet shards")
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--val_csv", default="data/val.csv")
    ap.add_argument("--skip_audio", action="store_true",
                    help="Skip audio hashing (much faster; exact-duplicate count unavailable)")
    args = ap.parse_args()

    if not args.raw_dir:
        ap.error("--raw_dir is required (or set NEYSHEKAR_RAW_DIR)")

    df = load_raw(args.raw_dir, args.skip_audio)
    problems = []

    print("\n" + "=" * 74)
    print(" 1. RAW DATASET")
    print("=" * 74)
    problems.append(row("raw records", len(df), REPORT["raw_total"]))

    print("\n" + "=" * 74)
    print(" 2. TEXT ANOMALIES IN THE RAW TRANSCRIPTS")
    print("=" * 74)
    problems.append(row("rows containing Arabic characters",
                        int(df["text_raw"].apply(lambda t: bool(ARABIC.search(t))).sum()),
                        REPORT["arabic_rows"]))
    problems.append(row("rows containing numeric digits",
                        int(df["text_raw"].apply(lambda t: bool(DIGITS.search(t))).sum()),
                        REPORT["digit_rows"]))

    print("\n" + "=" * 74)
    print(" 3. AUDIO DURATION (Whisper's valid 1s-30s window)")
    print("=" * 74)
    dur = pd.to_numeric(df["duration"], errors="coerce")
    problems.append(row("files shorter than 1.0s", int((dur < 1.0).sum()), REPORT["under_1s"]))
    problems.append(row("files longer than 30.0s", int((dur > 30.0).sum()), REPORT["over_30s"]))
    problems.append(row("files with unreadable duration", int(dur.isna().sum()), REPORT["corrupted"]))
    problems.append(row("max duration (s)", float(dur.max()), REPORT["max_duration"],
                        fmt="{:.2f}", tol=0.01))

    # ---- replay the cleaning pipeline exactly as data_prep.py does ----
    print("\n" + "=" * 74)
    print(" 4. REPLAYING THE CLEANING PIPELINE")
    print("=" * 74)
    valid = df[(dur >= 1.0) & (dur <= 30.0) & dur.notna()].copy().reset_index(drop=True)
    print(f"  after duration filter                        {len(valid):>12,}")

    valid["normalized_text"] = (
        valid["text_raw"].apply(apply_step1_normalization)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )

    if args.skip_audio:
        print("  exact duplicate removal                          (skipped: --skip_audio)")
        stage1 = valid
    else:
        dup_mask = valid.duplicated(subset=["audio_hash", "text_raw"], keep="first")
        problems.append(row("exact duplicates (same audio AND text)",
                            int(dup_mask.sum()), REPORT["exact_dups"]))
        stage1 = valid[~dup_mask].copy().reset_index(drop=True)

    stage1["fp"] = stage1["normalized_text"].apply(fingerprint)
    excess = 0
    for _, idx in stage1.groupby("fp").groups.items():
        if len(idx) > MAX_COPIES_PER_LONG_TEXT:
            if not is_short(stage1.loc[idx[0], "normalized_text"]):
                excess += len(idx) - MAX_COPIES_PER_LONG_TEXT
    problems.append(row("long-text excess copies dropped", excess, REPORT["long_excess_dropped"]))

    final_n = len(stage1) - excess
    problems.append(row("FINAL cleaned records", final_n, REPORT["final_total"]))

    # ---- confirm the committed CSVs match what we just recomputed ----
    print("\n" + "=" * 74)
    print(" 5. COMMITTED CSVs (what Task 2 actually trains on)")
    print("=" * 74)
    try:
        tr = pd.read_csv(args.train_csv)
        va = pd.read_csv(args.val_csv)
        csv_n = len(tr) + len(va)
        problems.append(row("rows in train.csv + val.csv", csv_n, final_n))
        print(f"  {'train / val split':<44} {len(tr):>6,} / {len(va):,}")
        ids = set(tr["id"]).union(va["id"])
        problems.append(row("unique ids", len(ids), csv_n))
        problems.append(row("train/val id overlap", len(set(tr['id']) & set(va['id'])), 0))
        all_txt = pd.concat([tr["cleaned_text"], va["cleaned_text"]]).astype(str)
        problems.append(row("Arabic chars left in labels",
                            int(all_txt.apply(lambda t: bool(ARABIC.search(t))).sum()), 0))
        problems.append(row("digits left in labels",
                            int(all_txt.apply(lambda t: bool(DIGITS.search(t))).sum()), 0))
        d2 = pd.to_numeric(pd.concat([tr["duration"], va["duration"]]), errors="coerce")
        row("mean duration (s)", float(d2.mean()), REPORT["mean_duration"], fmt="{:.2f}", tol=0.01)
    except FileNotFoundError as e:
        print(f"  [skipped] {e}")

    print("\n" + "=" * 74)
    failed = problems.count(False)
    if failed:
        print(f" RESULT: {failed} figure(s) disagree with the report -- correct them before submitting.")
    else:
        print(" RESULT: every checked figure reproduces the report exactly.")
    print("=" * 74)


if __name__ == "__main__":
    main()
