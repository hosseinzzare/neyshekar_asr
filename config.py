"""
Root launcher for the configuration module.

src/config.py holds the real definitions; this file exists only so that `python config.py` and
`from config import ...` work from the repository root, which is where the README tells people
to run everything.

The re-export is a star import on purpose. It used to be a hand-written list of names, and the
list had already fallen behind: src/config.py defined 46 module-level constants and this file
re-exported 45, silently omitting MAX_STEPS. Nothing imported it through here yet, so nothing
broke — but a shim whose whole job is to mirror another module should not need maintaining every
time that module gains a line. A star import cannot drift.
"""
from src.config import *          # noqa: F401,F403
from src.config import Config, set_seed, SEED  # noqa: F401  (explicit: used below)

if __name__ == '__main__':
    set_seed(SEED)
    print("Configuration loaded successfully via root config.py:")
    print(f"  - Model: {MODEL_NAME_OR_PATH}")          # noqa: F405
    print(f"  - Language: {LANGUAGE}")                 # noqa: F405
    print(f"  - Output Dir: {OUTPUT_DIR}")             # noqa: F405
    print(f"  - QLoRA Rank (R): {LORA_R}, Alpha: {LORA_ALPHA}")   # noqa: F405
