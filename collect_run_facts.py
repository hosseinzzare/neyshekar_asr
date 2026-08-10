"""
Collect Run Facts
=================

Pulls the numbers quoted in the Task 2 and Task 3 write-ups straight from the published
training artefacts, so every figure in the documentation has a traceable source rather than
being copied from a terminal window that no longer exists.

Reads four things from the model repository on the Hub:

  trainer_state.json     the full log history -- evaluation points, timings, best checkpoint
  all_results.json       final aggregate metrics
  train.log              the run's stdout, which carries the lines the Trainer never records:
                         trainable parameter count, peak VRAM, preflight check total
  ablation_logs.tar.gz   the controlled efficiency runs

Nothing here is recomputed. The point is to quote the run rather than to reconstruct it.

Usage
-----
    python collect_run_facts.py
    python collect_run_facts.py --repo hosseinzr/neyshekar-whisper-large-v3-lora
"""

import argparse
import io
import json
import re
import sys
import tarfile
import urllib.request

BASE = "https://huggingface.co/{repo}/resolve/main/{name}"


def fetch(repo, name, binary=False):
    url = BASE.format(repo=repo, name=name)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        return data if binary else data.decode("utf-8", errors="replace")
    except Exception as e:                                   # noqa: BLE001
        print(f"  [could not fetch {name}: {e}]")
        return None


def hms(seconds):
    s = int(seconds)
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def section(title):
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def report_state(repo):
    raw = fetch(repo, "trainer_state.json")
    if not raw:
        return
    st = json.loads(raw)
    hist = st.get("log_history", [])
    evals = [h for h in hist if "eval_wer" in h]
    trains = [h for h in hist if "loss" in h and "eval_loss" not in h]

    section("EVALUATION POINTS  (from trainer_state.json)")
    print(f"  {'step':>7}{'epoch':>8}{'val loss':>11}{'WER %':>9}{'CER %':>8}{'eval secs':>11}")
    print("  " + "-" * 62)
    for e in evals:
        print(f"  {e.get('step', 0):>7}{e.get('epoch', 0):>8.3f}"
              f"{e.get('eval_loss', float('nan')):>11.5f}"
              f"{e.get('eval_wer', float('nan')):>9.2f}{e.get('eval_cer', float('nan')):>8.2f}"
              f"{e.get('eval_runtime', float('nan')):>11.1f}")

    if len(evals) > 1:
        gaps = {evals[i]["step"] - evals[i - 1]["step"] for i in range(1, len(evals))}
        print(f"\n  evaluation interval(s) actually used : {sorted(gaps)}")
    print(f"  logging interval                     : "
          f"{trains[1]['step'] - trains[0]['step'] if len(trains) > 1 else '?'} steps")
    print(f"  total optimizer steps                : {st.get('global_step', '?'):,}")
    print(f"  epochs completed                     : {st.get('epoch', '?')}")
    print(f"  best checkpoint                      : {st.get('best_model_checkpoint', '?')}")
    print(f"  best metric (WER)                    : {st.get('best_metric', '?')}")
    tot_eval = sum(e.get("eval_runtime", 0) for e in evals)
    print(f"  time spent in in-training evaluation  : {hms(tot_eval)}  "
          f"({len(evals)} evaluations)")


def report_results(repo):
    raw = fetch(repo, "all_results.json")
    if not raw:
        return
    r = json.loads(raw)
    section("TOTALS  (from all_results.json)")
    tr, steps = r.get("train_runtime", 0), 6270
    print(f"  train_runtime            {tr:>12,.0f} s   = {hms(tr)}")
    print(f"  final eval runtime       {r.get('final_runtime', 0):>12,.0f} s   "
          f"= {hms(r.get('final_runtime', 0))}")
    print(f"  wall-clock per step      {tr / steps:>12.2f} s   "
          f"(train_runtime / {steps:,} steps, includes in-training evaluation)")
    print(f"  train_steps_per_second   {r.get('train_steps_per_second', 0):>12.3f}"
          f"     = {1 / r.get('train_steps_per_second', 1):.2f} s per step")
    print(f"  final WER / CER          {r.get('final_wer', 0):>12.2f} % / "
          f"{r.get('final_cer', 0):.2f} %")
    print(f"  final loss               {r.get('final_loss', 0):>12.5f}")
    print(f"  total FLOPs              {r.get('total_flos', 0):>12.3e}")


# Lines worth pulling out of stdout: the Trainer never puts these in trainer_state.json.
# Matching is case-insensitive and substring-based. The first version used fullmatch with
# [Pp]eak and silently missed "[PEAK VRAM] ..." because the tag is upper case -- which is
# exactly the kind of quiet miss this whole script exists to prevent.
LOG_PATTERNS = [
    ("trainable parameters", r"trainable params"),
    ("peak VRAM", r"peak vram|max_memory|peak memory"),
    ("preflight total", r"checks passed|preflight"),
    ("gradient checkpointing", r"gradient checkpointing"),
    ("precision", r"\[precision\]"),
    ("schedule", r"\[schedule\]|eval every"),
    ("GPU", r"\[gpu\]"),
    ("audio preprocessing", r"\[audio prep\]"),
    ("decode failures", r"decode.{0,20}fail|undecodable|dropped"),
    ("dataset match", r"\[match\]"),
    ("split sizes", r"\[split\]"),
    ("label alignment", r"\[verify\]"),
]


def report_log(repo):
    raw = fetch(repo, "train.log")
    if not raw:
        return
    lines = raw.splitlines()
    section(f"KEY LINES FROM train.log  ({len(lines):,} lines)")
    for label, pattern in LOG_PATTERNS:
        rx = re.compile(pattern, re.I)
        seen, uniq = set(), []
        for l in lines:
            l = l.strip()
            if rx.search(l) and l not in seen:
                seen.add(l)
                uniq.append(l)
        print(f"\n  --- {label} ---")
        if not uniq:
            print("      (no matching line)")
        for h in uniq[:6]:
            print(f"      {h[:160]}")


def report_ablation(repo):
    blob = fetch(repo, "ablation_logs.tar.gz", binary=True)
    if not blob:
        return
    section("ABLATION LOGS  (ablation_logs.tar.gz)")

    # The per-run numbers that matter: whether it fitted in memory at all, how long an
    # optimizer step took with evaluation excluded, and how much VRAM it peaked at.
    WANT = re.compile(
        r"outofmemory|out of memory|"
        r"train_runtime|eval_runtime|"
        r"peak vram|"
        r"gradient checkpointing\]|"
        r"\[precision\]|\[gpu\]|"
        r"s/step|s/eval-sample", re.I)

    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            print(f"  archive contains {len(members)} file(s): "
                  f"{', '.join(m.name for m in members)}")
            for m in members:
                print(f"\n  --- {m.name}  ({m.size:,} bytes) ---")
                body = tf.extractfile(m).read().decode("utf-8", errors="replace")
                if m.name.endswith(".json"):
                    print("      " + json.dumps(json.loads(body), indent=2).replace("\n", "\n      "))
                    continue
                seen, uniq = set(), []
                for l in body.splitlines():
                    l = l.strip()
                    if WANT.search(l) and l not in seen:
                        seen.add(l)
                        uniq.append(l)
                for l in uniq[:14]:
                    print(f"      {l[:170]}")
                if not uniq:
                    print("      (nothing matched; last lines follow)")
                    for l in [x for x in body.splitlines() if x.strip()][-6:]:
                        print(f"      {l.strip()[:170]}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  [could not read archive: {e}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="hosseinzr/neyshekar-whisper-large-v3-lora")
    args = p.parse_args()
    print(f"reading artefacts from https://huggingface.co/{args.repo}")
    report_results(args.repo)
    report_state(args.repo)
    report_log(args.repo)
    report_ablation(args.repo)
    print()


if __name__ == "__main__":
    sys.exit(main())
