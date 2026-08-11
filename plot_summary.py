"""
One figure for the top of the README.

The repository has five documents and around thirty pages. Someone opening it for the first time
needs the result before they will read any of that, and a table of numbers is not read in the
same glance a chart is. This draws the two things that actually answer "did it work":

  left   what fine-tuning bought, against the untouched base model on identical audio
  right  how the error rate moved during the run, and where it stopped moving

Every value is hard-coded from the published artefacts rather than recomputed, and each is
annotated below with where it came from, so the figure cannot drift away from the documents
without someone editing this file.

    python plot_summary.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig8_summary.png")

# run_artifacts/error_analysis_zeroshot.csv and error_analysis.csv, same 200 validation
# utterances, scored with src/metrics.normalize_persian_for_eval
BASE = {"WER": 36.97, "CER": 8.80, "exact": 23}
TUNED = {"WER": 8.74, "CER": 2.14, "exact": 115}

# run_artifacts/train.log, the five in-training evaluations over the fixed 500-row subset
STEPS = [1500, 3000, 4500, 6000, 6270]
WER = [16.30, 12.50, 9.57, 8.19, 8.17]
CER = [4.77, 3.12, 2.28, 1.94, 1.90]
FINAL_WER = 8.05          # full 5,900-row split, [FINAL EVAL] line in train.log

INK, MUTED, LINE = "#1c1917", "#6b645c", "#d9d4cd"
BASE_C, TUNED_C, CER_C = "#b4341f", "#0f7b52", "#8a5a2b"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis="y", color=LINE, linewidth=.7, alpha=.7)
    ax.set_axisbelow(True)


fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.1), dpi=170)
fig.patch.set_facecolor("white")

# ---- left: before and after, same audio -------------------------------------------------
labels = ["Word error rate", "Character error rate"]
x = np.arange(len(labels))
w = 0.34
a.bar(x - w / 2, [BASE["WER"], BASE["CER"]], w, label="Base model", color=BASE_C, alpha=.85)
a.bar(x + w / 2, [TUNED["WER"], TUNED["CER"]], w, label="After fine-tuning", color=TUNED_C)

for xi, (lo, hi) in zip(x, [(BASE["WER"], TUNED["WER"]), (BASE["CER"], TUNED["CER"])]):
    a.text(xi - w / 2, lo + .8, f"{lo:.2f}%", ha="center", fontsize=9.5, color=BASE_C, weight="bold")
    a.text(xi + w / 2, hi + .8, f"{hi:.2f}%", ha="center", fontsize=9.5, color=TUNED_C, weight="bold")

a.set_xticks(x)
a.set_xticklabels(labels, fontsize=10, color=INK)
a.set_ylabel("%", color=MUTED, fontsize=9)
a.set_ylim(0, 43)
a.set_title("Same 200 validation utterances, adapter the only difference",
            fontsize=10.5, color=INK, pad=10, loc="left")
a.legend(frameon=False, fontsize=9.5, labelcolor=MUTED, loc="upper right")
a.text(0.5, 30.5, f"exact transcriptions\n{BASE['exact']}  →  {TUNED['exact']}  of 200",
       ha="center", fontsize=9.5, color=MUTED, linespacing=1.5)
style(a)

# ---- right: the run ---------------------------------------------------------------------
b.plot(STEPS, WER, "o-", color=TUNED_C, linewidth=2, markersize=5, label="WER")
b.plot(STEPS, CER, "o-", color=CER_C, linewidth=2, markersize=5, label="CER")
b.axhline(FINAL_WER, color=TUNED_C, linestyle=":", linewidth=1.3, alpha=.8)
b.text(1120, FINAL_WER - 1.05, f"{FINAL_WER}%  —  full 5,900-row split",
       fontsize=8.5, color=TUNED_C, ha="left")

for s in (2090, 4180):
    b.axvline(s, color=LINE, linestyle="--", linewidth=1)
b.text(2090, 11.6, "epoch 1 | 2", fontsize=8, color=MUTED, ha="center",
       bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
b.text(4180, 11.6, "epoch 2 | 3", fontsize=8, color=MUTED, ha="center",
       bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))

for s, v, dx, dy, ha in ((STEPS[0], WER[0], 8, 10, "left"), (STEPS[-1], WER[-1], 4, 14, "center")):
    b.annotate(f"{v:.2f}%", (s, v), textcoords="offset points", xytext=(dx, dy),
               ha=ha, fontsize=9, color=TUNED_C, weight="bold")

b.set_xlabel("optimizer step", color=MUTED, fontsize=9)
b.set_ylabel("%", color=MUTED, fontsize=9)
b.set_ylim(0, 18.6)
b.set_xlim(1050, 6750)
b.set_title("3 epochs, 6,270 steps, single NVIDIA L4", fontsize=10.5, color=INK, pad=10, loc="left")
b.legend(frameon=False, fontsize=9.5, labelcolor=MUTED)
style(b)

fig.tight_layout(pad=1.6)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("written:", OUT)
