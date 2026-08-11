"""
End-to-end check of the inference container.

Why this exists: requirements-docker.txt originally pinned transformers 4.46.3, which could not
run this code at all — from_pretrained(dtype=...) is the 5.x signature. The pins were corrected
to transformers 5.14.1 / peft 0.20.0 / accelerate 1.14.0, and a dependency change on the
inference path is exactly the kind of thing that can silently alter model output. This script
posts the five committed validation clips to the running container and compares the transcripts
against what the previous build produced, so "the container still works" is a measurement rather
than an assumption.

Standard library only, so it runs anywhere Python does. No requests, no pip install.

Usage
-----
    python test_docker.py                       # expects the container on :8000
    python test_docker.py --url http://localhost:8000 --timeout 900

Exit code 0 if every clip matches, 1 otherwise.
"""

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

# What the container produced before the dependency pins were corrected. These are transcripts,
# not references: the point of the comparison is that changing transformers did not change what
# the model emits. Three of the five differ from the reference transcript, and those differences
# are analysed in demo.html and in Task 4.
EXPECTED = {
    "val_16322.wav": "فرض کنید برای اولین ماراتن خود تمرین می‌کنید.",
    "val_17098.wav": "جدی خیلی هم عالی",
    "val_31021.wav": "خیلی کمتر از آن یکی",
    "val_33008.wav": "کمرمان خم شد زیر این‌همه فشار.",
    "val_37678.wav": "آراد یه دانشمند شده بود و سینا یه ماجراجویی معرف.",
}
CLIPS_DIR = "test_clips"


def post_file(url, path, timeout=600):
    """Multipart POST without the requests library."""
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        payload = fh.read()

    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        payload,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Content-Length": str(len(body))})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_health(base, timeout):
    """The model loads after the port opens, so /health answers 503 until it is ready."""
    print(f"  waiting for {base}/health ", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
                if r.status == 200:
                    print(f"\n  ready after {int(timeout - (deadline - time.time()))}s")
                    print("  " + r.read().decode("utf-8")[:200] + "\n")
                    return True
        except urllib.error.HTTPError as e:
            if e.code != 503:
                print(f"\n  unexpected status {e.code}")
        except Exception:
            pass          # container still starting, or port not open yet
        print(".", end="", flush=True)
        time.sleep(10)
    print(f"\n  gave up after {timeout}s")
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--timeout", type=int, default=1200,
                   help="seconds to wait for the model to load (first run downloads 3 GB)")
    args = p.parse_args()
    base = args.url.rstrip("/")

    if not wait_for_health(base, args.timeout):
        sys.exit("Model never became ready. Check `docker logs asr`.")

    missing = [c for c in EXPECTED if not os.path.exists(os.path.join(CLIPS_DIR, c))]
    if missing:
        sys.exit(f"Missing clips in {CLIPS_DIR}/: {missing}\n"
                 f"Run this from the repository root.")

    print(f"  {'clip':<16}{'result':<10}{'audio s':>9}{'compute s':>11}")
    print("  " + "-" * 46)
    failures, rows = [], []
    for clip, expected in EXPECTED.items():
        path = os.path.join(CLIPS_DIR, clip)
        try:
            out = post_file(f"{base}/transcribe", path)
        except Exception as e:
            print(f"  {clip:<16}{'ERROR':<10}  {e}")
            failures.append((clip, "request failed", str(e)))
            continue
        got = out.get("text", "").strip()
        ok = got == expected
        rows.append((clip, out.get("audio_seconds"), out.get("compute_seconds")))
        print(f"  {clip:<16}{'match' if ok else 'DIFFERS':<10}"
              f"{out.get('audio_seconds', 0):>9.2f}{out.get('compute_seconds', 0):>11.2f}")
        if not ok:
            failures.append((clip, expected, got))

    audio = sum(r[1] or 0 for r in rows)
    comp = sum(r[2] or 0 for r in rows)
    print("  " + "-" * 46)
    print(f"  {'total':<16}{'':<10}{audio:>9.2f}{comp:>11.2f}")
    if audio:
        print(f"\n  {comp / audio:.1f}x slower than real time on this machine")

    if failures:
        print(f"\n  {len(failures)} clip(s) differ from the previous build:\n")
        for clip, expected, got in failures:
            print(f"  {clip}")
            print(f"    before : {expected}")
            print(f"    now    : {got}\n")
        print("  A difference here means the dependency change altered model output, and the\n"
              "  figures in the documents would need re-checking. Report it rather than\n"
              "  updating EXPECTED to match.\n")
        sys.exit(1)

    print("\n  All five transcripts identical to the previous build. "
          "The dependency change is safe.\n")


if __name__ == "__main__":
    main()
