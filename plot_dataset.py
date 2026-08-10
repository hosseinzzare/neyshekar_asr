"""
Dataset Distribution Plots (Task 1)
===================================

Produces the distribution figures for the Task 1 write-up and prints the statistics quoted
alongside them, so the numbers in the report and the numbers in the figures cannot drift apart.

Reads the cleaned dataset that Task 2 actually trains on, not the raw shards. The raw
distribution is not the interesting one -- the report's claims are about the corpus after
cleaning, and that is the corpus the model saw.

Figures
-------
  fig5_duration_hist.png     audio duration, with Whisper's 30 s window for scale
  fig6_transcript_length.png transcript length in characters and in words
  fig7_speech_rate.png       duration against transcript length, with the low-CPS region marked

The third figure is not requested by the brief. It is included because the claim that ~28% of
files carry substantial silence is the finding that justifies adding silence trimming to the
pipeline, and a scatter makes the case in a way a summary statistic cannot.

Usage
-----
    python plot_dataset.py
    python plot_dataset.py --csv data/neyshekar_cleaned.csv --outdir figures
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED, GREEN, ORANGE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#888888"

WHISPER_WINDOW_S = 30.0     # fixed input window of the feature extractor
MIN_DURATION_S = 1.0        # floor applied during cleaning
CPS_SUSPICIOUS = 8.0        # characters per second below which audio is mostly silence


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    dur = np.array([float(r["duration"]) for r in rows])
    txt = [r["cleaned_text"] for r in rows]
    chars = np.array([len(t) for t in txt])
    words = np.array([len(t.split()) for t in txt])
    return dur, chars, words


def stat_box(ax, lines, loc="upper right"):
    """
    Summary statistics as a corner box rather than rotated labels on the vertical lines.

    The first version wrote each statistic sideways next to its own line. With the median at
    5.04 s and the mean at 5.69 s those labels landed on top of each other and on the bars
    behind them. A box keeps the numbers legible and leaves the distribution unobstructed.
    """
    ax.legend(handles=[plt.Line2D([], [], linestyle="none", label=t) for t in lines],
              loc=loc, fontsize=8.5, handlelength=0, handletextpad=0,
              framealpha=0.92, borderpad=0.7)


def fig_duration(dur, outdir):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.hist(dur, bins=np.arange(0, 28.5, 0.5), color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, WHISPER_WINDOW_S)
    for x, colour in ((float(np.median(dur)), RED),
                      (float(dur.mean()), GREEN),
                      (float(np.percentile(dur, 99)), ORANGE)):
        ax.axvline(x, color=colour, ls="--", lw=1.4, zorder=5)

    ax.axvspan(0, MIN_DURATION_S, color=RED, alpha=0.12, zorder=0)
    # White background behind the label: it necessarily sits over the tallest part of the
    # histogram, since the region it points at is at the left edge where the bars begin.
    ax.annotate(f"< {MIN_DURATION_S:.0f}s removed (10 files)",
                xy=(MIN_DURATION_S, ax.get_ylim()[1] * 0.30), fontsize=8, color=RED,
                xytext=(18, 0), textcoords="offset points", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.axvline(WHISPER_WINDOW_S, color=GREY, lw=1.5)

    stat_box(ax, [
        f"median  {np.median(dur):.2f} s",
        f"mean    {dur.mean():.2f} s",
        f"p99     {np.percentile(dur, 99):.2f} s",
        f"max     {dur.max():.2f} s",
        "",
        f"Whisper window  {WHISPER_WINDOW_S:.0f} s",
        "nothing is truncated",
    ])

    ax.set_xlabel("Audio duration (seconds)")
    ax.set_ylabel("Number of files")
    ax.set_title(f"Audio duration distribution — {len(dur):,} cleaned recordings, "
                 f"{dur.sum() / 3600:.1f} hours total")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig5_duration_hist.png", dpi=150)
    plt.close(fig)


def fig_transcript(chars, words, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    axes[0].hist(chars, bins=np.arange(0, 205, 5), color=BLUE, edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Transcript length (characters)")
    axes[0].set_title("Characters")
    axes[0].axvline(float(np.median(chars)), color=RED, ls="--", lw=1.4)
    stat_box(axes[0], [f"median  {np.median(chars):.0f}", f"mean    {chars.mean():.1f}",
                       f"p99     {np.percentile(chars, 99):.0f}", f"max     {chars.max()}"])

    axes[1].hist(words, bins=np.arange(0, 42, 1), color=GREEN, edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("Transcript length (words)")
    axes[1].set_title("Words")
    axes[1].axvline(float(np.median(words)), color=RED, ls="--", lw=1.4)
    stat_box(axes[1], [f"median  {np.median(words):.0f}", f"mean    {words.mean():.1f}",
                       f"p99     {np.percentile(words, 99):.0f}", f"max     {words.max()}"])

    for ax in axes:
        ax.set_ylabel("Number of transcripts")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Transcript length distribution — both far below Whisper's 448-token label limit",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig6_transcript_length.png", dpi=150)
    plt.close(fig)


def fig_speech_rate(dur, chars, outdir):
    cps = chars / dur
    low = cps < CPS_SUSPICIOUS

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    axes[0].scatter(dur[~low], chars[~low], s=2, alpha=0.15, color=BLUE, label="normal speech rate")
    axes[0].scatter(dur[low], chars[low], s=2, alpha=0.15, color=RED,
                    label=f"CPS < {CPS_SUSPICIOUS:.0f}  ({low.sum():,}, {100 * low.mean():.1f}%)")
    xs = np.linspace(1, dur.max(), 50)
    axes[0].plot(xs, CPS_SUSPICIOUS * xs, color="black", lw=1.5, ls="--",
                 label=f"{CPS_SUSPICIOUS:.0f} characters per second")
    axes[0].set_xlabel("Audio duration (seconds)")
    axes[0].set_ylabel("Transcript length (characters)")
    axes[0].set_title("Duration against transcript length")
    leg = axes[0].legend(fontsize=8, markerscale=4, loc="upper left")
    for h in leg.legend_handles:
        try:
            h.set_alpha(1)
        except AttributeError:
            pass
    axes[0].grid(alpha=0.25)

    axes[1].hist(cps, bins=np.arange(0, 25, 0.4), color=BLUE, edgecolor="white", linewidth=0.4)
    axes[1].axvline(CPS_SUSPICIOUS, color=RED, lw=1.6, ls="--")
    axes[1].annotate(f"{low.sum():,} files ({100 * low.mean():.1f}%)\nbelow {CPS_SUSPICIOUS:.0f} CPS",
                     xy=(CPS_SUSPICIOUS, axes[1].get_ylim()[1] * 0.8), color=RED, fontsize=9,
                     ha="right", xytext=(-8, 0), textcoords="offset points")
    axes[1].set_xlabel("Characters per second")
    axes[1].set_ylabel("Number of files")
    axes[1].set_title(f"Speech rate — median {np.median(cps):.2f} CPS")
    axes[1].grid(alpha=0.25, axis="y")

    fig.suptitle("Points below the dashed line are short transcripts over long audio — "
                 "the signature of leading and trailing silence", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig7_speech_rate.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/neyshekar_cleaned.csv")
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dur, chars, words = load(args.csv)
    cps = chars / dur
    n = len(dur)

    fig_duration(dur, args.outdir)
    fig_transcript(chars, words, args.outdir)
    fig_speech_rate(dur, chars, args.outdir)

    def line(label, arr, fmt="{:.2f}"):
        v = [arr.mean(), np.median(arr), arr.std(), arr.min(), arr.max(),
             np.percentile(arr, 95), np.percentile(arr, 99)]
        print(f"  {label:<24}" + "".join(fmt.format(x).rjust(10) for x in v))

    print(f"\n{'=' * 96}\n  TASK 1 DISTRIBUTION SUMMARY — {n:,} cleaned records\n{'=' * 96}")
    print(f"  {'':<24}{'mean':>10}{'median':>10}{'std':>10}{'min':>10}{'max':>10}{'p95':>10}{'p99':>10}")
    print("  " + "-" * 94)
    line("duration (s)", dur)
    line("transcript (chars)", chars, "{:.1f}")
    line("transcript (words)", words, "{:.1f}")
    line("speech rate (CPS)", cps)

    print(f"\n  total audio                {dur.sum() / 3600:>8.2f} hours")
    print(f"  files below {CPS_SUSPICIOUS:.0f} CPS         {int((cps < CPS_SUSPICIOUS).sum()):>8,}"
          f"   ({100 * (cps < CPS_SUSPICIOUS).mean():.2f}%)")
    print(f"  longest recording          {dur.max():>8.2f} s   "
          f"({100 * dur.max() / WHISPER_WINDOW_S:.0f}% of Whisper's window)")
    print(f"  longest transcript         {chars.max():>8,} characters   "
          f"(well inside the 448-token label limit)")
    print(f"\n  Figures written to {args.outdir}/\n")


if __name__ == "__main__":
    main()
