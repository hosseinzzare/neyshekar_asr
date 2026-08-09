"""
Error Analysis (Task 4)
=======================

Runs the fine-tuned model over validation samples, compares each prediction with its
reference, and writes everything to a CSV sorted worst-first so the 20 required examples can
be picked and categorised offline.

Uses the project's own preprocessing (src.dataset.prepare_dataset) rather than a
reimplementation, so the audio the model sees here is identical to what it saw in training.

Also emits a rough automatic category for each error -- omission, insertion, substitution,
number, repetition -- as a starting point. The categories still need a human read, but they
turn "here are 20 wrong sentences" into "here are 20 wrong sentences, grouped".

Usage
-----
    # on a machine where the dataset is already cached (fastest)
    python analyze_errors.py --n_samples 300

    # on a fresh machine (Colab): download only what is needed
    python analyze_errors.py --n_samples 200 --max_shards 2

    # adapter from the Hub instead of a local directory
    python analyze_errors.py --adapter hosseinzr/neyshekar-whisper-large-v3-lora
"""

import argparse
import csv
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig  # noqa: E402
from peft import PeftModel  # noqa: E402

import config  # noqa: E402
from dataset import load_custom_dataset, prepare_dataset  # noqa: E402
from metrics import normalize_persian_for_eval  # noqa: E402

DIGIT_RE = re.compile(r"[\d۰-۹٠-٩]")


def per_sample_scores(ref, hyp):
    """WER and CER for a single pair, on normalised text (same rule as training)."""
    import jiwer
    r, h = normalize_persian_for_eval(ref), normalize_persian_for_eval(hyp)
    if not r.strip():
        return None
    wer = jiwer.wer(r, h)
    cer = jiwer.cer(r, h)
    out = jiwer.process_words(r, h)
    return {
        "wer": wer, "cer": cer,
        "sub": out.substitutions, "ins": out.insertions, "dele": out.deletions,
        "ref_words": len(r.split()), "hyp_words": len(h.split()),
        "ref_norm": r, "hyp_norm": h,
    }


def categorise(s):
    """Rough first-pass label. A human still has to read these, but it groups them."""
    tags = []
    if s["dele"] > s["ins"] and s["dele"] > s["sub"]:
        tags.append("omitted-words")
    if s["ins"] > s["dele"] and s["ins"] > s["sub"]:
        tags.append("inserted-words")
    if s["sub"] >= max(s["ins"], s["dele"]) and s["sub"] > 0:
        tags.append("substitution")
    if DIGIT_RE.search(s["hyp_norm"]):
        tags.append("digits-in-output")          # Task 1 lexicalised all numbers to words
    words = s["hyp_norm"].split()
    if len(words) > 3 and max((words.count(w) for w in set(words)), default=0) >= 4:
        tags.append("repetition-loop")
    if s["hyp_words"] > 2 * max(s["ref_words"], 1):
        tags.append("hallucination-long")
    if s["hyp_words"] == 0:
        tags.append("empty-output")
    return "|".join(tags) if tags else "other"


def main():
    p = argparse.ArgumentParser(description="Generate predictions and rank them by error")
    p.add_argument("--adapter", default="./whisper-large-v3-neyshekar-qlora",
                   help="LoRA adapter: local directory or Hub repo id")
    p.add_argument("--n_samples", type=int, default=300,
                   help="how many validation samples to transcribe")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_shards", type=int, default=None,
                   help="limit the dataset download (use on a machine without the cache)")
    p.add_argument("--out", default="error_analysis.csv")
    p.add_argument("--no_quantization", action="store_true",
                   help="load the base model in fp16 instead of 4-bit (needs ~7 GB VRAM)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible -- generation on CPU is impractically slow.")
    print(f"[GPU] {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    # ---- data (same preprocessing as training) ----
    ds = load_custom_dataset(max_shards=args.max_shards)["validation"]
    n = min(args.n_samples, len(ds))
    ds = ds.shuffle(seed=config.SEED).select(range(n))
    print(f"[DATA] {n:,} validation samples selected (seed={config.SEED})")

    # ---- model ----
    processor = WhisperProcessor.from_pretrained(
        config.MODEL_NAME_OR_PATH, language=config.LANGUAGE, task=config.TASK)

    if args.no_quantization:
        base = WhisperForConditionalGeneration.from_pretrained(
            config.MODEL_NAME_OR_PATH, dtype=torch.float16, device_map="auto")
    else:
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True)
        base = WhisperForConditionalGeneration.from_pretrained(
            config.MODEL_NAME_OR_PATH, quantization_config=qc, device_map="auto")

    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    model.generation_config.language = config.LANGUAGE
    model.generation_config.task = config.TASK
    model.generation_config.forced_decoder_ids = None
    print(f"[MODEL] adapter loaded from {args.adapter}")

    # ---- transcribe ----
    rows = []
    for start in range(0, n, args.batch_size):
        batch = ds.select(range(start, min(start + args.batch_size, n)))
        feats, refs, ids = [], [], []
        for ex in batch:
            out = prepare_dataset(dict(ex), processor)
            if not out.get("_decode_ok", True):
                continue
            feats.append(out["input_features"])
            refs.append(str(ex["cleaned_text"]))
            ids.append(ex["id"])
        if not feats:
            continue

        x = torch.tensor(np.stack(feats)).to(model.device).half()
        with torch.no_grad():
            gen = model.generate(input_features=x, max_new_tokens=225)
        hyps = processor.batch_decode(gen, skip_special_tokens=True)

        for i, r, h in zip(ids, refs, hyps):
            s = per_sample_scores(r, h)
            if s is None:
                continue
            rows.append({"id": i, "reference": r, "prediction": h.strip(),
                         "wer": round(s["wer"], 4), "cer": round(s["cer"], 4),
                         "substitutions": s["sub"], "insertions": s["ins"],
                         "deletions": s["dele"], "ref_words": s["ref_words"],
                         "hyp_words": s["hyp_words"], "category": categorise(s)})
        done = min(start + args.batch_size, n)
        print(f"  {done}/{n} transcribed", end="\r", flush=True)

    # ---- report ----
    rows.sort(key=lambda r: (-r["wer"], -r["cer"]))
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    perfect = sum(1 for r in rows if r["wer"] == 0)
    print(f"\n\n{'=' * 70}")
    print(f"  {len(rows):,} samples transcribed")
    print(f"  {perfect:,} exactly correct ({100 * perfect / len(rows):.1f}%)")
    print(f"  {len(rows) - perfect:,} contain at least one error")
    print(f"  corpus WER {sum(r['wer'] for r in rows) / len(rows):.4f} (unweighted mean)")
    print(f"{'=' * 70}")

    cats = {}
    for r in rows:
        if r["wer"] > 0:
            for t in r["category"].split("|"):
                cats[t] = cats.get(t, 0) + 1
    print("\n  first-pass error categories:")
    for t, c in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"    {t:<22} {c:>5}")

    print(f"\n  worst 20 written to {args.out} (sorted worst-first)\n")
    for r in rows[:20]:
        print(f"  WER {r['wer']:.2f} | {r['category']}")
        print(f"    ref : {r['reference'][:90]}")
        print(f"    hyp : {r['prediction'][:90]}")


if __name__ == "__main__":
    main()
