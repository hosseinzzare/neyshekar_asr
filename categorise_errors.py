"""
Error Categorisation (Task 4)
=============================

Takes the CSV written by analyze_errors.py and sorts the errors into linguistically meaningful
groups, rather than the substitution/deletion/insertion counts that edit distance produces.

Two decisions shape this script.

First, everything is scored on normalised text, using the same normalize_persian_for_eval that
computed the reported WER. The CSV stores raw strings, so comparing them directly counts a
missing full stop as a word error and inflates the rate well above the figure quoted for the
model. Importing the training-time normaliser rather than reimplementing it keeps one
definition of "the same string" across the project.

Second, categories are assigned per word, not per utterance. A sentence usually contains
several unrelated problems -- a spacing difference here, a genuine misrecognition there -- and
labelling the whole sentence forces every mixed case into the worst bucket. Counting word
events shows what the model actually gets wrong and in what proportion.

Word-level events
-----------------
  split               one reference word came out as several  (صداوسیما -> صدا و سیما)
  merge               several reference words came out as one  (بلا گردان -> بلاگردان)
  homophone-letter    differs only by letters Persian pronounces identically (غ/ع, ز/ذ/ض/ظ)
  final-consonant     the word lost or gained its last character
  substitution        a genuine word-level recognition error
  insertion           a word with no counterpart in the reference
  deletion            a reference word with no counterpart in the output

split, merge and homophone-letter are grouped as "orthographic": the transcription is
phonetically right and differs only in spelling convention, which no amount of audio can
resolve.

Usage
-----
    python categorise_errors.py --csv error_analysis.csv
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from metrics import normalize_persian_for_eval
except ImportError:                                    # standalone copy of the CSV, no repo
    def normalize_persian_for_eval(text):
        if not isinstance(text, str):
            return ""
        text = text.replace("ي", "ی").replace("ك", "ک")
        text = re.sub(r"[ً-ٰٟ]", "", text)
        text = re.sub(r"[^\w\s‌]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    print("[warn] src/metrics.py not importable; using a local copy of the normaliser")

ZWNJ = "‌"

# Letters Persian writes differently but pronounces the same. A model working from audio has no
# signal to choose between them; the reference spelling is a convention, not a sound.
#
# ع and غ are NOT such a pair and were wrongly grouped here at first. In Persian غ is a velar
# fricative that has merged with ق, while ع is a glottal stop that is usually not realised at
# all — غلامان and علامان do not sound alike, so a model confusing them has made a real
# recognition error, not chosen a different spelling. Removing that group moved one event out of
# the orthographic column and into substitution.
HOMOPHONE_GROUPS = [set("اآ"), set("تط"), set("ثسص"), set("حه"),
                    set("ذزضظ"), set("قغ"), set("ئی"), set("ۀه")]

ORTHOGRAPHIC = {"split", "merge", "zwnj-placement", "attached-preposition", "homophone-letter"}


def fold_homophones(s):
    out = []
    for ch in s:
        for g in HOMOPHONE_GROUPS:
            if ch in g:
                ch = sorted(g)[0]
                break
        out.append(ch)
    return "".join(out)


def classify_pair(a, b):
    """
    Label a single aligned reference/hypothesis word pair.

    Order matters. The zero-width non-joiner test has to run before the homophone test,
    otherwise "یخبندان" against "یخ‌بندان" is folded to the same string and reported as a
    homophone confusion when the only difference is where a joiner was placed.
    """
    an, bn = a.replace(ZWNJ, ""), b.replace(ZWNJ, "")

    if an == bn:
        return "zwnj-placement"

    # The preposition "به" is often written fused to the following noun with its ه dropped --
    # "بحساب" for "به‌حساب", "براستی" for "به‌راستی". Both spellings are read identically, so
    # this belongs with the orthographic differences rather than with recognition errors.
    if {an, bn} == {"ب" + an[1:], "به" + an[1:]} or {an, bn} == {"ب" + bn[1:], "به" + bn[1:]}:
        return "attached-preposition"

    if fold_homophones(an) == fold_homophones(bn):
        return "homophone-letter"
    if a[:-1] == b or b[:-1] == a:
        return "final-consonant"
    return "substitution"


def align(rw, hw, span=4):
    """
    Greedy left-to-right alignment that recognises one-to-many and many-to-one matches.

    jiwer aligns strictly one word to one word, so a word the model split in two is reported as
    a substitution plus an insertion and the split itself never appears in the counts. Since
    splits and merges are the specific thing under investigation here, the alignment has to be
    able to see them: before falling back to a one-to-one substitution, the scan checks whether
    joining up to `span` words on either side produces an exact match.

    Greedy rather than optimal. It can mis-align after a long run of unrelated errors, but on
    utterances of this length it agrees with the jiwer totals closely enough, and it is short
    enough to read and check by hand -- which matters more here than squeezing out the last
    fraction of alignment accuracy.
    """
    i = j = 0
    events = []
    while i < len(rw) and j < len(hw):
        if rw[i] == hw[j]:
            i, j = i + 1, j + 1
            continue

        matched = False
        for k in range(2, span + 1):                       # one reference word -> k output words
            if j + k <= len(hw) and "".join(hw[j:j + k]) == rw[i].replace(ZWNJ, ""):
                events.append(("split", rw[i], " ".join(hw[j:j + k])))
                i, j, matched = i + 1, j + k, True
                break
        if matched:
            continue
        for k in range(2, span + 1):                       # k reference words -> one output word
            if i + k <= len(rw) and "".join(rw[i:i + k]) == hw[j].replace(ZWNJ, ""):
                events.append(("merge", " ".join(rw[i:i + k]), hw[j]))
                i, j, matched = i + k, j + 1, True
                break
        if matched:
            continue

        # a word present on one side only, recognised by the next word lining up again
        if j + 1 < len(hw) and i < len(rw) and hw[j + 1] == rw[i]:
            events.append(("insertion", "", hw[j]))
            j += 1
            continue
        if i + 1 < len(rw) and j < len(hw) and rw[i + 1] == hw[j]:
            events.append(("deletion", rw[i], ""))
            i += 1
            continue

        events.append((classify_pair(rw[i], hw[j]), rw[i], hw[j]))
        i, j = i + 1, j + 1

    events += [("deletion", w, "") for w in rw[i:]]
    events += [("insertion", "", w) for w in hw[j:]]
    return events


def corpus_rates(refs, hyps):
    """Corpus-level WER and CER under strict and ZWNJ-relaxed tokenisation."""
    try:
        import jiwer
    except ImportError:
        print("  [skip] pip install jiwer for corpus-level rates\n")
        return
    relax = lambda s: re.sub(r"\s+", " ", s.replace(ZWNJ, " ")).strip()

    print(f"{'policy':<32}{'WER':>9}{'CER':>9}{'S':>7}{'D':>6}{'I':>6}")
    print("-" * 72)
    rates = {}
    for name, rr, hh in (("strict (ZWNJ preserved)", refs, hyps),
                         ("relaxed (ZWNJ -> space)", [relax(x) for x in refs],
                          [relax(x) for x in hyps])):
        m = jiwer.process_words(rr, hh)
        rates[name] = m.wer
        print(f"{name:<32}{100 * m.wer:>8.2f}%{100 * jiwer.cer(rr, hh):>8.2f}%"
              f"{m.substitutions:>7}{m.deletions:>6}{m.insertions:>6}")
    print(f"\n  ZWNJ normalisation alone is worth "
          f"{100 * (rates['strict (ZWNJ preserved)'] - rates['relaxed (ZWNJ -> space)']):.2f} "
          f"WER points\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="error_analysis.csv")
    p.add_argument("--out", default="error_categories.csv")
    p.add_argument("--top", type=int, default=8, help="examples printed per category")
    args = p.parse_args()

    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["ref_norm"] = normalize_persian_for_eval(r["reference"])
        r["hyp_norm"] = normalize_persian_for_eval(r["prediction"])
    rows = [r for r in rows if r["ref_norm"]]

    exact = [r for r in rows if r["ref_norm"] == r["hyp_norm"]]
    errs = [r for r in rows if r["ref_norm"] != r["hyp_norm"]]

    print(f"\n{'=' * 72}")
    print(f"  {len(rows)} samples · {len(errs)} with at least one error "
          f"({100 * len(errs) / len(rows):.1f}%) · {len(exact)} exact after normalisation")
    print(f"{'=' * 72}\n")

    corpus_rates([r["ref_norm"] for r in rows], [r["hyp_norm"] for r in rows])

    # ---- word-level events -------------------------------------------------------------
    events, per_row = [], {}
    for r in errs:
        ev = align(r["ref_norm"].split(), r["hyp_norm"].split())
        ev = [e for e in ev if e[0] != "match"]
        per_row[r["id"]] = ev
        events += ev

    counts = Counter(e[0] for e in events)
    n_ref_words = sum(len(r["ref_norm"].split()) for r in rows)

    print(f"{'=' * 72}\n  WORD-LEVEL ERROR EVENTS\n{'=' * 72}")
    print(f"{'event':<20}{'count':>8}{'% of events':>14}{'% of ref words':>17}")
    print("-" * 72)
    for k, n in counts.most_common():
        print(f"{k:<20}{n:>8}{100 * n / len(events):>13.1f}%{100 * n / n_ref_words:>16.2f}%")
    orth = sum(counts[k] for k in ORTHOGRAPHIC)
    print("-" * 72)
    print(f"{'orthographic':<20}{orth:>8}{100 * orth / len(events):>13.1f}%"
          f"{100 * orth / n_ref_words:>16.2f}%")
    print(f"{'genuine':<20}{len(events) - orth:>8}"
          f"{100 * (len(events) - orth) / len(events):>13.1f}%"
          f"{100 * (len(events) - orth) / n_ref_words:>16.2f}%")

    # ---- utterance-level view ----------------------------------------------------------
    pure = sum(1 for ev in per_row.values() if ev and all(e[0] in ORTHOGRAPHIC for e in ev))
    mixed = sum(1 for ev in per_row.values()
                if any(e[0] in ORTHOGRAPHIC for e in ev)
                and any(e[0] not in ORTHOGRAPHIC for e in ev))
    print(f"\n{'=' * 72}\n  UTTERANCES WITH ERRORS ({len(errs)})\n{'=' * 72}")
    print(f"  {pure:>4}  ({100 * pure / len(errs):>4.1f}%)  orthographic differences only")
    print(f"  {mixed:>4}  ({100 * mixed / len(errs):>4.1f}%)  a mix of orthographic and genuine")
    print(f"  {len(errs) - pure - mixed:>4}  "
          f"({100 * (len(errs) - pure - mixed) / len(errs):>4.1f}%)  genuine errors only")

    # ---- examples ----------------------------------------------------------------------
    for cat, _ in counts.most_common():
        sel = [e for e in events if e[0] == cat][:args.top]
        print(f"\n{'=' * 72}\n  {cat.upper()}  ({counts[cat]})\n{'=' * 72}")
        for _, a, b in sel:
            print(f"    {a or '—':<38} ->  {b or '—'}")

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "event", "reference_span", "hypothesis_span",
                    "orthographic", "full_reference", "full_prediction"])
        for r in errs:
            for kind, a, b in per_row[r["id"]]:
                w.writerow([r["id"], kind, a, b, int(kind in ORTHOGRAPHIC),
                            r["ref_norm"], r["hyp_norm"]])
    print(f"\nWritten to {args.out}\n")


if __name__ == "__main__":
    main()
