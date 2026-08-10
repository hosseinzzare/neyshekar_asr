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
    "max_duration": 27.18,
    "mean_duration": 5.69,
    # Recomputed from the raw shards rather than copied from the first exploratory run: the
    # notebook that produced the original 10,983 measured at a slightly earlier pipeline stage.
    # The report quotes whatever this script reproduces, so the two can never drift apart.
    "low_cps": 11036,
    "clipped": 8693,
    "short_dup_preserved": 3479,
    "long_repeats_preserved": 18520,
    "long_repeats_capped_kept": 1191,
}

# Thresholds the Task 1 report used when flagging suspicious samples.
CPS_SUSPICIOUS = 8.0        # characters per second below which audio is mostly silence
CLIP_CEILING = 0.99         # peak amplitude at or above which a file is treated as clipped

ARABIC = re.compile(r"[\u064a\u0643\u0629\u0623\u0625\u0671\u0624\u0626\u0649]")
DIGITS = re.compile(r"[\d\u06f0-\u06f9\u0660-\u0669]")

MAX_COPIES_PER_LONG_TEXT = 3


def fingerprint(text: str) -> str:
    """Identical to data_prep.py: strip diacritics, punctuation, spaces and ZWNJ."""
    return remove_spaces_and_zwnj(remove_diacritics(text))


def is_short(text: str) -> bool:
    """data_prep.py's exemption rule for high-frequency short conversational phrases."""
    return (len(text) < 15) or (len(text.split()) < 4)


def peak_amplitude(blob) -> float:
    """
    Peak absolute amplitude of one audio blob, or NaN if it cannot be decoded.

    Used only to re-count clipped files. Reads from the in-memory bytes rather than a
    temporary file so nothing touches disk.
    """
    import io
    import soundfile as sf
    try:
        data, _ = sf.read(io.BytesIO(blob["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return float(np.abs(data).max()) if data.size else 0.0
    except Exception:
        return float("nan")


def load_raw(raw_dir: str, skip_audio: bool, check_clipping: bool = False) -> pd.DataFrame:
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

    need_audio = (not skip_audio) or check_clipping
    cols = ["id", "text", "duration"] + (["audio"] if need_audio else [])
    frames = []
    for i, path in enumerate(files, 1):
        df = pd.read_parquet(path, columns=cols)
        if need_audio:
            # Derive everything needed from the bytes here, then drop them, so peak memory
            # stays at one shard rather than the whole 7 GB corpus.
            if not skip_audio:
                df["audio_hash"] = df["audio"].apply(
                    lambda a: hashlib.md5(a["bytes"]).hexdigest()
                    if isinstance(a, dict) and a.get("bytes") is not None
                    else str(a)
                )
            if check_clipping:
                df["peak_amp"] = df["audio"].apply(peak_amplitude)
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
    ap.add_argument("--check_clipping", action="store_true",
                    help="Decode every waveform to re-count clipped files. Slow (~10 min) "
                         "because it decodes all 40k files; everything else needs only metadata.")
    args = ap.parse_args()

    if not args.raw_dir:
        ap.error("--raw_dir is required (or set NEYSHEKAR_RAW_DIR)")

    df = load_raw(args.raw_dir, args.skip_audio, args.check_clipping)
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
    drop_idx = []
    for _, idx in stage1.groupby("fp").groups.items():
        if len(idx) > MAX_COPIES_PER_LONG_TEXT:
            if not is_short(stage1.loc[idx[0], "normalized_text"]):
                # data_prep.py keeps the first MAX_COPIES_PER_LONG_TEXT rows in file order.
                drop_idx.extend(list(idx)[MAX_COPIES_PER_LONG_TEXT:])
    excess = len(drop_idx)
    problems.append(row("long-text excess copies dropped", excess, REPORT["long_excess_dropped"]))

    # Split the surviving long-text repeats into the two groups the report distinguishes:
    # sentences that occurred 2-3 times on their own, and sentences that occurred more often
    # and were cut back to the cap. Both are "preserved", but only the second involved a
    # deliberate loss of acoustic variety, so they are worth reporting separately.
    natural_2_3, retained_from_capped = 0, 0
    for _, idx in stage1.groupby("fp").groups.items():
        if is_short(stage1.loc[idx[0], "normalized_text"]):
            continue
        if 2 <= len(idx) <= MAX_COPIES_PER_LONG_TEXT:
            natural_2_3 += len(idx)
        elif len(idx) > MAX_COPIES_PER_LONG_TEXT:
            retained_from_capped += MAX_COPIES_PER_LONG_TEXT
    problems.append(row("long repeats preserved (2-3 copies)", natural_2_3,
                        REPORT["long_repeats_preserved"]))
    problems.append(row("long repeats kept after capping", retained_from_capped,
                        REPORT["long_repeats_capped_kept"]))

    # Materialise the final set rather than only counting it. The suspicious-sample figures
    # below are quoted in the report as percentages of the 39,332 records that training
    # actually sees, so they have to be measured on that set -- computing them on stage1
    # silently includes the 532 rows that were about to be dropped.
    final_df = stage1.drop(index=drop_idx).reset_index(drop=True)
    final_n = len(final_df)
    problems.append(row("FINAL cleaned records", final_n, REPORT["final_total"]))

    # Short conversational phrases are exempt from the duplicate cap. The report quotes this
    # figure inside the duplicate analysis, so it counts short rows that ARE duplicated,
    # not every short row in the corpus.
    fp_counts = stage1["fp"].value_counts()
    short_dup = int(sum(
        1 for t, k in zip(stage1["normalized_text"], stage1["fp"])
        if is_short(t) and fp_counts[k] > 1
    ))
    problems.append(row("short duplicated phrases preserved", short_dup,
                        REPORT["short_dup_preserved"]))

    # ---- suspicious samples: the two findings that drove the Task 2 preprocessing ----
    print("\n" + "=" * 74)
    print(" 5. SUSPICIOUS SAMPLES (thresholds as used in the report)")
    print("=" * 74)
    cps = final_df["normalized_text"].str.len() / pd.to_numeric(final_df["duration"], errors="coerce")
    n_low = int((cps < CPS_SUSPICIOUS).sum())
    problems.append(row(f"characters per second < {CPS_SUSPICIOUS}", n_low, REPORT["low_cps"]))
    print(f"  {'  as a share of the final set':<44} {100 * n_low / final_n:>11.2f}%")
    print(f"  {'CPS mean / median':<44} {cps.mean():>6.2f} / {cps.median():.2f}")

    if args.check_clipping:
        peak = pd.to_numeric(final_df["peak_amp"], errors="coerce")
        n_clip = int((peak >= CLIP_CEILING).sum())
        problems.append(row(f"peak amplitude >= {CLIP_CEILING} (clipped)",
                            n_clip, REPORT["clipped"]))
        print(f"  {'  as a share of the final set':<44} {100 * n_clip / final_n:>11.2f}%")
        print(f"  {'files that failed to decode':<44} {int(peak.isna().sum()):>12,}")
    else:
        print("  clipped-file count                               (skipped: pass --check_clipping)")

    # ---- confirm the committed CSVs match what we just recomputed ----
    print("\n" + "=" * 74)
    print(" 6. COMMITTED CSVs (what Task 2 actually trains on)")
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
        row("max duration in committed CSVs (s)", float(d2.max()), REPORT["max_duration"],
            fmt="{:.2f}", tol=0.01)

        # Splitting on id keeps the same recording out of both halves, but says nothing about
        # the SENTENCES. Where a transcript recurs across speakers, the same text can land in
        # both splits with different audio. That is not leakage in the usual sense -- no
        # validation waveform was trained on -- but the decoder's language prior has seen the
        # sentence, so validation WER reads better than it would on unseen text. Reported here
        # because the figure belongs in the answer to "are there duplicated transcripts".
        tr_fps = set(tr["cleaned_text"].astype(str).map(fingerprint))
        va_fp = va["cleaned_text"].astype(str).map(fingerprint)
        shared = va_fp.isin(tr_fps)
        va_long = va["cleaned_text"].astype(str).map(lambda t: not is_short(t))
        print(f"\n  {'val transcripts also seen in train':<44} "
              f"{int(shared.sum()):>12,}  ({100 * shared.mean():.1f}% of val)")
        print(f"  {'  of which are long transcripts':<44} "
              f"{int((shared & va_long).sum()):>12,}  "
              f"({100 * (shared & va_long).mean():.1f}% of val)")
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
