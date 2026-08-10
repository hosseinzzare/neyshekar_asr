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
    print(f"  {'step':>7}{'epoch':>8}{'val loss':>11}{'WER %':>9}{'CER %':>8}"
          f"{'eval secs':>11}{'samples':>10}")
    print("  " + "-" * 72)
    for e in evals:
        # eval_samples_per_second x eval_runtime recovers how many rows were actually scored.
        # The subset size is not recorded directly, and it decides whether the in-training
        # numbers are comparable with the final full-split figure.
        n = e.get("eval_samples_per_second", 0) * e.get("eval_runtime", 0)
        print(f"  {e.get('step', 0):>7}{e.get('epoch', 0):>8.3f}"
              f"{e.get('eval_loss', float('nan')):>11.5f}"
              f"{e.get('eval_wer', float('nan')):>9.2f}{e.get('eval_cer', float('nan')):>8.2f}"
              f"{e.get('eval_runtime', float('nan')):>11.1f}{n:>10.0f}")

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


STEPS_PER_EPOCH = 2090
EVAL_STEPS = (1500, 3000, 4500, 6000, 6270)


def report_curve(repo):
    """
    Every statistic the Task 3 analysis quotes about the training curve.

    Read from training_curve.csv rather than recomputed from the model, so the figures in the
    write-up, the figures in the plots and the figures here all come from one file.
    """
    raw = fetch(repo, "training_curve.csv")
    if not raw:
        return
    import csv
    rows = list(csv.DictReader(io.StringIO(raw)))
    step = [int(r["step"]) for r in rows]
    loss = [float(r["train_loss"]) for r in rows]
    lr = [float(r["learning_rate"]) for r in rows]
    gn = [None if r["grad_norm"] in ("", "nan", "NaN") else float(r["grad_norm"]) for r in rows]

    section("TRAINING CURVE  (from training_curve.csv)")
    print(f"  logged points                  {len(rows):>12,}")
    print(f"  logging interval               {step[1] - step[0]:>12} steps")
    print(f"  train loss  first / last       {loss[0]:>12.4f} / {loss[-1]:.4f}")
    print(f"  train loss  min / max          {min(loss):>12.4f} / {max(loss):.4f}")

    print("\n  --- early collapse (the first phase of the decline) ---")
    for target in (10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000):
        i = min(range(len(step)), key=lambda k: abs(step[k] - target))
        print(f"      step {step[i]:>5}   loss {loss[i]:.4f}   lr {lr[i]:.3e}")

    print("\n  --- training loss at each evaluation step ---")
    print(f"      {'step':>7}{'exact':>10}{'mean of 10 pts':>17}")
    for s in EVAL_STEPS:
        i = min(range(len(step)), key=lambda k: abs(step[k] - s))
        lo, hi = max(0, i - 9), i + 1
        window = loss[lo:hi]
        print(f"      {step[i]:>7}{loss[i]:>10.4f}{sum(window) / len(window):>17.4f}")

    print("\n  --- mean training loss per epoch ---")
    for e in (1, 2, 3):
        lo, hi = STEPS_PER_EPOCH * (e - 1), STEPS_PER_EPOCH * e
        vals = [l for s, l in zip(step, loss) if lo < s <= hi]
        print(f"      epoch {e}   mean {sum(vals) / len(vals):.4f}   "
              f"({len(vals)} points, steps {lo + 1}-{hi})")

    # The plotted curve steps down sharply at each epoch boundary rather than declining
    # smoothly. Averaging the 200 steps either side of the boundary measures that step, and
    # separates it from the within-epoch trend it would otherwise be confused with.
    print("\n  --- behaviour at the epoch boundaries ---")
    for e in (1, 2):
        b = STEPS_PER_EPOCH * e
        before = [l for s, l in zip(step, loss) if b - 200 < s <= b]
        after = [l for s, l in zip(step, loss) if b < s <= b + 200]
        mb, ma = sum(before) / len(before), sum(after) / len(after)
        print(f"      boundary at step {b}:  last 200 steps of epoch {e} mean {mb:.4f}"
              f"   ->  first 200 of epoch {e + 1} mean {ma:.4f}"
              f"   (drop {mb - ma:.4f}, {100 * (mb - ma) / mb:.1f}%)")
    for e in (1, 2, 3):
        lo, hi = STEPS_PER_EPOCH * (e - 1), STEPS_PER_EPOCH * e
        vals = [(s, l) for s, l in zip(step, loss) if lo < s <= hi]
        first200 = [l for s, l in vals if s <= lo + 200]
        last200 = [l for s, l in vals if s > hi - 200]
        f, la = sum(first200) / len(first200), sum(last200) / len(last200)
        print(f"      within epoch {e}: first 200 steps mean {f:.4f}  ->  "
              f"last 200 mean {la:.4f}   (decline {f - la:.4f})")

    good = [g for g in gn if g is not None]
    nan_steps = [s for s, g in zip(step, gn) if g is None]
    print("\n  --- gradient norm ---")
    print(f"      min {min(good):.3f}   max {max(good):.3f}   mean {sum(good) / len(good):.3f}"
          f"   median {sorted(good)[len(good) // 2]:.3f}")
    print(f"      NaN / missing: {len(nan_steps)} of {len(gn)} logged points"
          f"{'  at step(s) ' + ', '.join(map(str, nan_steps[:8])) if nan_steps else ''}")

    # HuggingFace applies max_grad_norm=1.0 by default and train.py does not override it, so
    # the logged norm is the value BEFORE clipping. How often it exceeded 1.0 decides whether
    # "no clipping was needed" is a true statement about this run.
    over = sum(1 for g in good if g > 1.0)
    print(f"      above the 1.0 clipping threshold: {over} of {len(good)} "
          f"({100 * over / len(good):.1f}%)")
    for t in (1.5, 2.0, 2.5):
        print(f"      above {t}: {sum(1 for g in good if g > t)}")

    # Context around the single NaN: was it an isolated scaler event or part of a disturbance?
    for ns in nan_steps[:3]:
        i = step.index(ns)
        lo, hi = max(0, i - 3), min(len(step), i + 4)
        print(f"\n      around the NaN at step {ns}:")
        for k in range(lo, hi):
            g = "NaN" if gn[k] is None else f"{gn[k]:.3f}"
            print(f"        step {step[k]:>5}   loss {loss[k]:.4f}   grad norm {g}")

    print("\n  --- learning rate ---")
    peak_i = max(range(len(lr)), key=lambda k: lr[k])
    print(f"      first logged  {lr[0]:.3e}  at step {step[0]}")
    print(f"      peak          {lr[peak_i]:.3e}  at step {step[peak_i]}")
    print(f"      final         {lr[-1]:.3e}  at step {step[-1]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="hosseinzr/neyshekar-whisper-large-v3-lora")
    args = p.parse_args()
    print(f"reading artefacts from https://huggingface.co/{args.repo}")
    report_results(args.repo)
    report_state(args.repo)
    report_curve(args.repo)
    report_log(args.repo)
    report_ablation(args.repo)
    print()


if __name__ == "__main__":
    sys.exit(main())
