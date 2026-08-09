"""
Error Categorisation (Task 4)
=============================

Takes the CSV produced by analyze_errors.py and sorts the errors into linguistically
meaningful categories, rather than the generic substitution/insertion/deletion counts that
edit distance produces.

The motivating observation: inspecting the worst predictions shows that a large share are not
recognition failures at all but Persian word-segmentation differences — "همانگونه" versus
"همان گونه", "بحساب" versus "به‌حساب". Word error rate punishes these severely (splitting one
word into two costs a deletion plus two insertions) while character error rate barely moves.
That asymmetry explains why this model scores WER 8.17% but CER only 1.90%.

Categories, checked in order of specificity:

  segmentation-only   identical once spaces and ZWNJ are removed
  homophone-letter    differs only by letters that sound the same in Persian
  final-consonant     a word lost or gained its last character
  digits              a digit survived into the output (Task 1 lexicalised all numbers)
  repetition          a token repeats four or more times
  empty               nothing was produced
  length-mismatch     output is more than twice or less than half the reference length
  other-substitution  a genuine word-level recognition error

Usage
-----
    python categorise_errors.py                       # reads error_analysis.csv
    python categorise_errors.py --csv path.csv --top 20
"""

import argparse
import csv
import re
from collections import Counter

ZWNJ = "‌"
DIGITS = re.compile(r"[\d۰-۹٠-٩]")

# Persian letter groups that are pronounced identically. A model transcribing from audio
# alone cannot distinguish them; the correct choice is a spelling convention, not a sound.
HOMOPHONE_GROUPS = [
    set("اآ"),
    set("تط"),
    set("ثسص"),
    set("حه"),
    set("ذزضظ"),
    set("عغ"),
    set("قغ"),
    set("ئیي"),
    set("کك"),
]


def strip_spaces(s):
    return s.replace(" ", "").replace(ZWNJ, "").replace("‏", "").strip()


def canonical_homophones(s):
    """Map every homophone group to a single representative letter."""
    out = []
    for ch in s:
        for g in HOMOPHONE_GROUPS:
            if ch in g:
                ch = sorted(g)[0]
                break
        out.append(ch)
    return "".join(out)


def categorise(ref, hyp):
    r, h = ref.strip(), hyp.strip()
    if not h:
        return "empty-output"

    rs, hs = strip_spaces(r), strip_spaces(h)

    # 1. identical once spacing is ignored -> purely a segmentation difference
    if rs == hs:
        return "segmentation-only"

    # 2. identical once spacing AND homophone letters are normalised
    if canonical_homophones(rs) == canonical_homophones(hs):
        return "homophone-letter"

    # 3. a single word lost or gained its last character.
    #    Compared word by word rather than over the whole string: a dropped final consonant
    #    usually happens in the middle of an utterance ("برداشت زعفران آغاز شد" ->
    #    "برداشت زعفرا آغاز شد"), so checking only the last character of the full string
    #    misses almost every real instance.
    rws, hws = r.split(), h.split()
    if len(rws) == len(hws):
        diff = [(a, b) for a, b in zip(rws, hws) if a != b]
        if len(diff) == 1:
            a, b = diff[0]
            if a[:-1] == b or b[:-1] == a:
                return "final-consonant"

    if DIGITS.search(h):
        return "digits-in-output"

    words = h.split()
    if len(words) > 3 and max((words.count(w) for w in set(words)), default=0) >= 4:
        return "repetition-loop"

    rw, hw = len(r.split()), len(h.split())
    if hw > 2 * max(rw, 1) or hw < 0.5 * rw:
        return "length-mismatch"

    return "other-substitution"


def corpus_metrics(rows):
    """
    Corpus-level WER under two normalisation policies.

    Two things are being corrected here relative to the quick number printed by
    analyze_errors.py.

    First, that number was the unweighted mean of the per-sample rates, which lets a short
    utterance count as much as a long one. The figure reported for the model is corpus-level:
    total edits divided by total reference words. Both are printed below so the difference is
    visible rather than hidden.

    Second, the strict policy tokenises on whitespace only, so a zero-width non-joiner does not
    break a word. "وقتی‌که" is one token and "وقتی که" is two, and the mismatch is scored as a
    deletion plus two insertions even though the two spellings are pronounced identically. The
    relaxed policy converts ZWNJ to a space in both strings before scoring, which removes that
    particular penalty.

    The relaxed number is a diagnostic, not the headline. It does not fix the other half of the
    problem: "صداوسیما" against "صدا و سیما" has no ZWNJ to convert, and no rule can join or
    split those without a lexicon. The count of segmentation-only errors is the honest measure
    of that second family, and it is reported separately.
    """
    try:
        import jiwer
    except ImportError:
        print("  [skip] pip install jiwer to see corpus-level rates\n")
        return

    refs = [r["reference"] for r in rows]
    hyps = [r["prediction"] for r in rows]
    relaxed = lambda s: re.sub(r"\s+", " ", s.replace(ZWNJ, " ")).strip()

    strict = jiwer.process_words(refs, hyps)
    loose = jiwer.process_words([relaxed(x) for x in refs], [relaxed(x) for x in hyps])

    print(f"{'policy':<34}{'WER':>9}{'CER':>9}{'S':>7}{'D':>6}{'I':>6}")
    print("-" * 72)
    for name, m, rr, hh in (("strict (ZWNJ preserved)", strict, refs, hyps),
                            ("relaxed (ZWNJ -> space)", loose,
                             [relaxed(x) for x in refs], [relaxed(x) for x in hyps])):
        print(f"{name:<34}{100 * m.wer:>8.2f}%{100 * jiwer.cer(rr, hh):>8.2f}%"
              f"{m.substitutions:>7}{m.deletions:>6}{m.insertions:>6}")

    mean_wer = sum(float(r["wer"]) for r in rows) / len(rows)
    print(f"\n  corpus-level WER is the reported figure; the unweighted per-sample mean "
          f"is {100 * mean_wer:.2f}%")
    print(f"  ZWNJ normalisation alone accounts for "
          f"{100 * (strict.wer - loose.wer):.2f} WER points\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="error_analysis.csv")
    p.add_argument("--out", default="error_categories.csv")
    p.add_argument("--top", type=int, default=20, help="how many examples to print per category")
    args = p.parse_args()

    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    errors = [r for r in rows if float(r["wer"]) > 0]

    for r in errors:
        r["category"] = categorise(r["reference"], r["prediction"])

    counts = Counter(r["category"] for r in errors)

    print(f"\n{'=' * 72}")
    print(f"  {total} samples · {len(errors)} with at least one error "
          f"({100 * len(errors) / total:.1f}%) · {total - len(errors)} exact matches")
    print(f"{'=' * 72}\n")

    corpus_metrics(rows)
    print(f"{'category':<22}{'count':>7}{'% of errors':>13}{'mean WER':>11}")
    print("-" * 72)
    for cat, n in counts.most_common():
        mw = sum(float(r["wer"]) for r in errors if r["category"] == cat) / n
        print(f"{cat:<22}{n:>7}{100 * n / len(errors):>12.1f}%{mw:>11.3f}")

    # the headline number
    seg = counts.get("segmentation-only", 0) + counts.get("homophone-letter", 0)
    print("-" * 72)
    print(f"{'orthographic total':<22}{seg:>7}{100 * seg / len(errors):>12.1f}%")
    print("\n  'Orthographic' = the transcription is phonetically correct and differs only in")
    print("  spacing or in the choice between letters that sound identical in Persian.\n")

    # examples per category, worst first
    for cat, _ in counts.most_common():
        sel = sorted((r for r in errors if r["category"] == cat),
                     key=lambda r: -float(r["wer"]))[:args.top]
        print(f"\n{'=' * 72}\n  {cat.upper()}  ({counts[cat]} cases)\n{'=' * 72}")
        for r in sel:
            print(f"  WER {float(r['wer']):.2f}")
            print(f"    ref : {r['reference'][:100]}")
            print(f"    hyp : {r['prediction'][:100]}")

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
        w.writeheader()
        w.writerows(sorted(errors, key=lambda r: (r["category"], -float(r["wer"]))))
    print(f"\nWritten to {args.out}\n")


if __name__ == "__main__":
    main()
