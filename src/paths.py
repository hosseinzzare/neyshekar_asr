"""
Path Resolution for the Task 1 Investigation Scripts
====================================================

The raw Neyshekar dataset is 7.3 GB of audio and is not in this repository, so the scripts
that read it have to be told where it lives. That location used to be written into the top of
each script as a literal drive path, which meant the scripts ran on exactly one machine and
nowhere else.

Resolution order, first match wins:

  1. a --dataset_path / --output_dir argument on the command line
  2. the NEYSHEKAR_RAW_DIR / NEYSHEKAR_OUT_DIR environment variables
  3. for the output directory only, ./investigation_results relative to the working directory

There is no default for the dataset itself: guessing at it would fail later and less clearly
than saying so up front.

Usage
-----
    from paths import resolve_paths
    DATASET_PATH, OUTPUT_DIR = resolve_paths()

    $ python src/run_step1_investigation.py --dataset_path "D:/neyshekar/data"
    $ NEYSHEKAR_RAW_DIR=/mnt/data/neyshekar python src/run_step1_investigation.py
"""

import argparse
import os
import sys

DEFAULT_OUTPUT_DIR = "investigation_results"


def resolve_paths(description: str = "Neyshekar dataset investigation step"):
    """
    Return (dataset_path, output_dir), taking them from the command line or the environment.

    parse_known_args rather than parse_args so a caller that defines its own flags is not
    broken by this one, and so importing the module inside a notebook does not abort on the
    notebook's own argv.
    """
    p = argparse.ArgumentParser(description=description, add_help=False)
    p.add_argument("--dataset_path", default=os.environ.get("NEYSHEKAR_RAW_DIR"),
                   help="folder holding the raw train-*.parquet shards "
                        "(or set NEYSHEKAR_RAW_DIR)")
    p.add_argument("--output_dir",
                   default=os.environ.get("NEYSHEKAR_OUT_DIR", DEFAULT_OUTPUT_DIR),
                   help=f"where to write intermediates (default: ./{DEFAULT_OUTPUT_DIR})")
    args, _ = p.parse_known_args()

    if not args.dataset_path:
        sys.exit(
            "ERROR: the raw dataset location is not set.\n"
            "  This script reads the original parquet shards, which are too large to keep in\n"
            "  the repository. Point it at them with either:\n"
            "      --dataset_path \"<folder containing train-*.parquet>\"\n"
            "  or the environment variable NEYSHEKAR_RAW_DIR."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    return args.dataset_path, args.output_dir


def find_shards(dataset_path):
    """
    Return the sorted raw train-*.parquet shards under `dataset_path`.

    A HuggingFace snapshot puts the shards in a `data/` subdirectory, but a manual download
    often leaves them at the top level, and the four investigation scripts each globbed the top
    level only. Pointing any of them at the snapshot root -- the obvious thing to do, and what
    the README asks for -- therefore found nothing and exited claiming the dataset was missing.
    Checking both places is two lines and removes a class of error that looks like a missing
    dataset rather than a wrong path.

    Only train-*.parquet is matched, deliberately. A recursive sweep for *.parquet also picks up
    validation shards and any intermediate parquet an earlier step wrote, which would silently
    change the row counts every figure in Task 1 rests on.
    """
    import glob

    for candidate in (dataset_path, os.path.join(dataset_path, "data")):
        shards = sorted(glob.glob(os.path.join(candidate, "train-*.parquet")))
        if shards:
            return shards

    sys.exit(
        f"ERROR: no train-*.parquet shards under {dataset_path!r}.\n"
        f"  Looked in that folder and in its data/ subdirectory.\n"
        f"  Point --dataset_path or NEYSHEKAR_RAW_DIR at the folder holding the shards."
    )
