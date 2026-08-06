"""
Pre-flight Check
================

Run this immediately after cloning the repo on a new machine (Lightning.ai, Colab, ...)
and BEFORE starting a long training run.

It verifies two things:

  1. CODE  -- that every correctness fix is actually present in the checked-out source.
              Stale clones, half-synced Google Drive folders and leftover __pycache__ have
              all silently served OLD code during this project. Each check below maps to a
              specific bug that produced no error and no obviously wrong loss curve.

  2. DATA  -- that the committed Task 1 CSVs are intact (row counts, unique ids, no
              train/val leakage, fully normalised labels).

Stdlib only, runs in about a second, needs no GPU and no torch.

Usage
-----
    python preflight_check.py
    echo $?      # 0 = safe to train, 1 = do NOT start the run
"""

import csv
import os
import re
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

HERE = os.path.dirname(os.path.abspath(__file__))
ARABIC = re.compile(r"[يكةأإٱؤئى]")
DIGITS = re.compile(r"[\d۰-۹٠-٩]")

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    return ok


def read(rel):
    path = os.path.join(HERE, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def must_contain(src, needle):
    return src is not None and needle in src


def must_not_contain(src, needle):
    return src is not None and needle not in src


# ----------------------------------------------------------------------------------
print("\n" + "=" * 74)
print(" 1. CODE -- each check guards a bug that previously failed SILENTLY")
print("=" * 74)

ds = read("src/dataset.py")
check("src/dataset.py exists", ds is not None)

if ds:
    # Whisper's bos_token_id (50257) is NOT the token at position 0 of a label
    # (that is <|startoftranscript|>, 50258). Comparing against bos_token_id never
    # matched, so the SOT token was never stripped and the decoder prefix came out as
    # "<|sot|><|sot|><|fa|>..." -- corrupting teacher forcing with no error.
    check("SOT strip uses <|startoftranscript|>, not bos_token_id",
          must_contain(ds, 'convert_tokens_to_ids("<|startoftranscript|>")')
          and must_not_contain(ds, "decoder_start_token_id=processor.tokenizer.bos_token_id"),
          "duplicated <|sot|> in decoder input")

    # datasets>=3 returns a torchcodec AudioDecoder, which matched none of the old
    # branches -> every sample fell through to a ZERO waveform (training on silence).
    check("audio kept undecoded (Audio(decode=False))", must_contain(ds, "Audio(decode=False)"),
          "AudioDecoder object would not be recognised")
    check("torchcodec AudioDecoder handled as fallback", must_contain(ds, "get_all_samples"))
    check("no silent zero-waveform substitution", must_not_contain(ds, "np.zeros(int(16000"),
          "would train the model to transcribe silence")

    # Joining on transcript text is ambiguous: ~59% of rows share a duplicated transcript.
    check("audio<->label join uses the unique 'id' key", must_contain(ds, "train_id_map"))
    check("no set-membership text matching", must_not_contain(ds, "in all_texts"),
          "re-admits duplicates Task 1 removed")
    check("train/val leakage guard present", must_contain(ds, "TRAIN/VAL LEAKAGE"))
    check("label alignment probe present", must_contain(ds, "Label alignment OK"))

    # Falling back to the Hub's raw 'text' would silently train on UNCLEANED transcripts.
    check("labels must come from cleaned_text", must_contain(ds, '"cleaned_text" not in batch'),
          "could silently use raw uncleaned text")

    check("labels truncated to Whisper's 448 limit", must_contain(ds, "max_length=448"))
    check("partial-shard download supported", must_contain(ds, "verification_mode"))

    # Task 1 called for silence trimming and peak normalisation; they must run BEFORE the
    # log-Mel features are extracted, and trimming must run before normalisation so the peak
    # is measured on speech rather than on noise in the silence being discarded.
    check("silence trimming implemented", must_contain(ds, "def trim_silence("))
    check("peak normalisation implemented", must_contain(ds, "def peak_normalize("))
    if ds and "def trim_silence(" in ds and "inputs = processor.feature_extractor(" in ds:
        i_trim = ds.find("audio_array = trim_silence(")
        i_norm = ds.find("audio_array = peak_normalize(")
        i_feat = ds.find("inputs = processor.feature_extractor(")
        check("preprocessing order: trim -> normalise -> features",
              0 < i_trim < i_norm < i_feat,
              "wrong order would measure the peak on silence, or skip preprocessing entirely")

md = read("src/model.py")
check("generation language pinned to Persian",
      must_contain(md, "generation_config.language"),
      "large-v3 would auto-detect and decode some clips as Arabic/Urdu")
# prepare_model_for_kbit_training() is k-bit specific and force-enables gradient checkpointing.
check("quantization can be disabled", must_contain(md, "if use_quantization:"))
check("kbit prep only used for quantized models",
      must_contain(md, "model.gradient_checkpointing_enable()"),
      "unquantized path must enable checkpointing itself")
check("bf16 hardware support is checked", must_contain(md, "is_bf16_supported()"))

mt = read("src/metrics.py")
check("empty-reference guard in metrics (jiwer crash)",
      must_contain(mt, "empty reference"))
check("metrics do not mutate label_ids in place",
      must_contain(mt, "np.asarray(label_ids).copy()"))

tr = read("src/train.py")
for flag in ("--max_shards", "--max_eval_samples", "--final_full_eval", "--eval_steps",
             "--no_quantization", "--no_vad_trim", "--no_peak_norm"):
    check(f"train.py exposes {flag}", must_contain(tr, f'"{flag}"'))

# transformers raises if fp16 and bf16 are both enabled, and mixing bf16 weights with an
# fp16 autocast context can silently destabilise training.
check("fp16/bf16 chosen consistently with quantization",
      must_contain(tr, "fp16=use_fp16,") and must_contain(tr, "bf16=use_bf16,"),
      "precision flags must follow the quantization mode")

# `python train.py` puts the repo ROOT first on sys.path, so `import config` inside src/
# resolves to the ROOT launcher -- any name missing from its re-export list is an
# AttributeError at startup.
root_cfg = read("config.py")
src_cfg = read("src/config.py")
check("MAX_EVAL_SAMPLES defined in src/config.py", must_contain(src_cfg, "MAX_EVAL_SAMPLES = Config"))
check("MAX_EVAL_SAMPLES re-exported by root config.py", must_contain(root_cfg, "MAX_EVAL_SAMPLES"),
      "AttributeError before training starts")

# every config.X referenced in src/ must survive the root launcher's explicit import list
if root_cfg and src_cfg:
    used = set()
    for rel in ("src/dataset.py", "src/model.py", "src/metrics.py", "src/train.py"):
        s = read(rel) or ""
        used |= set(re.findall(r"(?<![.\w])config\.([A-Za-z_]\w*)", s))
    exported = set(re.findall(r"^\s{4}([A-Z_]\w*),?\s*$", root_cfg, re.M))
    exported |= set(re.findall(r"^\s{4}(set_seed|Config),?\s*$", root_cfg, re.M))
    missing = sorted(used - exported)
    check("all config.X names re-exported by root launcher", not missing, f"missing: {missing}")

# ----------------------------------------------------------------------------------
print("\n" + "=" * 74)
print(" 2. DATA -- Task 1 output integrity")
print("=" * 74)

tp, vp = os.path.join(HERE, "data", "train.csv"), os.path.join(HERE, "data", "val.csv")
if not (os.path.exists(tp) and os.path.exists(vp)):
    check("data/train.csv and data/val.csv present", False, "CSV labels missing from the clone")
else:
    check("data/train.csv and data/val.csv present", True)

    def load(p):
        with open(p, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    tr_rows, va_rows = load(tp), load(vp)
    check("train rows == 33,432", len(tr_rows) == 33432, f"got {len(tr_rows):,}")
    check("val rows == 5,900", len(va_rows) == 5900, f"got {len(va_rows):,}")

    tr_ids, va_ids = {int(r["id"]) for r in tr_rows}, {int(r["id"]) for r in va_rows}
    check("all ids unique", len(tr_ids) + len(va_ids) == len(tr_rows) + len(va_rows))
    check("no train/val id overlap", not (tr_ids & va_ids), f"{len(tr_ids & va_ids)} shared ids")

    labels = [r["cleaned_text"] for r in tr_rows + va_rows]
    check("no Arabic characters left in labels",
          sum(1 for t in labels if ARABIC.search(t)) == 0)
    check("no raw digits left in labels",
          sum(1 for t in labels if DIGITS.search(t)) == 0)
    check("no empty labels", all(t.strip() for t in labels))

# ----------------------------------------------------------------------------------
failed = [n for n, ok, _ in results if not ok]
print("\n" + "=" * 74)
if failed:
    print(f" RESULT: {len(failed)} CHECK(S) FAILED -- do NOT start the training run.")
    for n in failed:
        print(f"   - {n}")
    print("\n Most likely cause: a stale clone, an unpushed commit, or leftover __pycache__.")
    print(" Fix:  git pull  &&  find . -name __pycache__ -type d -exec rm -rf {} +")
else:
    print(f" RESULT: all {len(results)} checks passed. Safe to start training.")
print("=" * 74)

sys.exit(1 if failed else 0)
