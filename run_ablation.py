"""
Efficiency Ablation Runner
==========================

Runs the same short training job under several configurations, then prints a comparison of
speed and peak memory. The point is to answer -- with measurements rather than intuition --
which knobs actually reduce wall-clock time for a full 3-epoch run, and which merely trade
memory for nothing.

Design notes
------------
* Every configuration uses an IDENTICAL workload (same steps, same data, same eval size).
  Only the variable under test changes, so the comparison is meaningful.
* Batch size is always redistributed to keep the EFFECTIVE batch at 16, which is what the
  learning-rate recipe was validated for. Configurations that change the effective batch are
  marked, because their timings are not directly comparable.
* An out-of-memory failure is a RESULT, not a crash: it is recorded and the suite continues.
* Absolute timings only apply to the GPU that produced them. The transferable quantity is the
  RATIO between configurations on the same GPU -- see --reference_s_per_step.

Usage
-----
    python run_ablation.py                              # full suite
    python run_ablation.py --max_steps 25               # shorter/longer runs
    python run_ablation.py --only baseline,fp32         # a subset
    python run_ablation.py --reference_s_per_step 8.02  # project onto another GPU
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (description, extra CLI args, keeps effective batch 16?)
CONFIGS = {
    "baseline": (
        "4-bit NF4 + gradient checkpointing (current recipe)",
        [], True,
    ),
    "fp32": (
        "no quantization, fp32 weights + autocast",
        ["--no_quantization"], True,
    ),
    "baseline-b4": (
        "4-bit, batch 4 x accum 4 (memory-lean comparison pair)",
        ["--train_batch_size", "4", "--grad_accum", "4"], True,
    ),
    "fp32-b4": (
        "no quantization, batch 4 x accum 4 (memory-lean comparison pair)",
        ["--no_quantization", "--train_batch_size", "4", "--grad_accum", "4"], True,
    ),
    "no-gradckpt": (
        "4-bit, gradient checkpointing OFF (expected to OOM on <24 GB)",
        ["--no_gradient_checkpointing"], True,
    ),
}


def parse_metrics(output: str) -> dict:
    """Pull the timing/quality numbers out of a training log."""
    def num(pattern):
        m = re.search(pattern, output)
        return float(m.group(1)) if m else None

    return {
        "train_runtime": num(r"'train_runtime':\s*'?([\d.]+)'?"),
        "eval_runtime": num(r"'eval_runtime':\s*'?([\d.]+)'?"),
        "eval_wer": num(r"'eval_wer':\s*'?([\d.]+)'?"),
        "eval_cer": num(r"'eval_cer':\s*'?([\d.]+)'?"),
        "peak_vram": num(r"\[PEAK VRAM\]\s*([\d.]+)\s*GiB"),
        "oom": "OutOfMemoryError" in output,
    }


def run_one(name, extra_args, args):
    desc, cli, _ = CONFIGS[name]
    cmd = [
        sys.executable, "train.py",
        "--max_steps", str(args.max_steps),
        "--max_shards", str(args.max_shards),
        "--eval_steps", str(args.max_steps),      # exactly one eval, at the end
        "--save_steps", str(args.max_steps),
        "--max_eval_samples", str(args.max_eval_samples),
        "--output_dir", f"./ablation-{name}",
    ] + cli

    print(f"\n{'=' * 78}\n  {name}  --  {desc}\n{'=' * 78}")
    print("  $ " + " ".join(cmd[1:]))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                          timeout=args.timeout)
    wall = time.time() - t0
    out = proc.stdout + proc.stderr

    res = parse_metrics(out)
    res.update(name=name, description=desc, wall=wall, returncode=proc.returncode)

    if res["oom"]:
        print(f"  -> OUT OF MEMORY after {wall:.0f}s  (a result, not a failure)")
    elif proc.returncode != 0:
        print(f"  -> FAILED (exit {proc.returncode}) after {wall:.0f}s")
        tail = [l for l in out.strip().splitlines() if l.strip()][-6:]
        for line in tail:
            print("     " + line[:160])
    else:
        tr, ev = res["train_runtime"], res["eval_runtime"]
        if tr is not None and ev is not None:
            pure = tr - ev
            res["s_per_step"] = pure / args.max_steps
            res["s_per_eval_sample"] = ev / args.max_eval_samples
            print(f"  -> {res['s_per_step']:.2f} s/step | "
                  f"{res['s_per_eval_sample']:.2f} s/eval-sample | "
                  f"peak {res['peak_vram'] or float('nan'):.1f} GiB | "
                  f"WER {res['eval_wer']} CER {res['eval_cer']}")
        else:
            print(f"  -> completed but timings could not be parsed")
    return res


def main():
    p = argparse.ArgumentParser(description="Measure the speed impact of training options")
    p.add_argument("--max_steps", type=int, default=25)
    p.add_argument("--max_shards", type=int, default=2)
    p.add_argument("--max_eval_samples", type=int, default=100)
    p.add_argument("--timeout", type=int, default=3600, help="per-configuration timeout (s)")
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated subset of: " + ", ".join(CONFIGS))
    p.add_argument("--total_steps", type=int, default=6267,
                   help="optimizer steps in the real 3-epoch run, for projection")
    p.add_argument("--reference_s_per_step", type=float, default=None,
                   help="known s/step for the TARGET GPU under the baseline config; the measured "
                        "ratios are applied to it to project timings onto that GPU")
    p.add_argument("--out", type=str, default="ablation_results.json")
    args = p.parse_args()

    names = [n.strip() for n in args.only.split(",")] if args.only else list(CONFIGS)
    for n in names:
        if n not in CONFIGS:
            p.error(f"unknown configuration {n!r}; choose from: {', '.join(CONFIGS)}")

    print(f"Workload per configuration: {args.max_steps} steps, "
          f"{args.max_eval_samples} eval samples, {args.max_shards} shard(s)")

    results = []
    for n in names:
        try:
            results.append(run_one(n, CONFIGS[n][1], args))
        except subprocess.TimeoutExpired:
            print(f"  -> TIMEOUT after {args.timeout}s")
            results.append({"name": n, "description": CONFIGS[n][0], "timeout": True})

    # ---------------- summary ----------------
    print("\n\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print(f"{'config':<14}{'s/step':>9}{'vs base':>9}{'peak GiB':>10}{'WER':>8}{'CER':>8}")
    print("-" * 78)

    base = next((r for r in results if r.get("name") == "baseline" and r.get("s_per_step")), None)
    for r in results:
        if r.get("oom"):
            print(f"{r['name']:<14}{'OOM':>9}{'-':>9}{'-':>10}{'-':>8}{'-':>8}")
        elif r.get("timeout"):
            print(f"{r['name']:<14}{'TIMEOUT':>9}{'-':>9}{'-':>10}{'-':>8}{'-':>8}")
        elif r.get("s_per_step"):
            rel = f"{base['s_per_step'] / r['s_per_step']:.2f}x" if base else "-"
            print(f"{r['name']:<14}{r['s_per_step']:>9.2f}{rel:>9}"
                  f"{(r.get('peak_vram') or 0):>10.1f}{(r.get('eval_wer') or 0):>8.1f}"
                  f"{(r.get('eval_cer') or 0):>8.1f}")
        else:
            print(f"{r['name']:<14}{'FAILED':>9}{'-':>9}{'-':>10}{'-':>8}{'-':>8}")

    # ---------------- projection ----------------
    if base:
        print("\n" + "=" * 78)
        print(f"  PROJECTED FULL RUN ({args.total_steps:,} steps = 3 epochs)")
        print("=" * 78)
        ref = args.reference_s_per_step
        if ref:
            print(f"  Absolute timings below are projected onto a GPU whose BASELINE speed is")
            print(f"  {ref:.2f} s/step, by applying the speed ratios measured here.\n")
        print(f"{'config':<14}{'this GPU':>14}{'target GPU':>16}")
        print("-" * 78)
        for r in results:
            if not r.get("s_per_step"):
                continue
            here = args.total_steps * r["s_per_step"] / 3600
            if ref:
                ratio = base["s_per_step"] / r["s_per_step"]
                there = args.total_steps * (ref / ratio) / 3600
                print(f"{r['name']:<14}{here:>12.1f} h{there:>14.1f} h")
            else:
                print(f"{r['name']:<14}{here:>12.1f} h{'-':>16}")
        if not ref:
            print("\n  Pass --reference_s_per_step to project these onto another GPU.")

    with open(os.path.join(HERE, args.out), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
