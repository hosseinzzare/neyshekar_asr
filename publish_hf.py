"""
Push the run artefacts and the model card to the Hugging Face model repository.

The adapter itself was uploaded from the training machine. What was missing afterwards is the
evidence: the per-utterance CSVs behind every figure quoted in the documents, and the training
log. Without error_analysis_zeroshot.csv in particular, the headline claim of this project —
36.97% before fine-tuning against 8.74% after, on identical audio — cannot be checked by anyone
but me, which makes it an assertion rather than a result.

Authentication
--------------
Never put the token in this file or on the command line, where it lands in shell history.
Either log in once:

    hf auth login

or set it for the session only:

    $env:HF_TOKEN = "hf_..."        # PowerShell
    export HF_TOKEN="hf_..."        # bash

The token needs write access to the model repository.

Usage
-----
    python publish_hf.py                 # show what would be uploaded, change nothing
    python publish_hf.py --push          # upload the artefacts
    python publish_hf.py --push --card   # artefacts and the model card
"""

import argparse
import os
import sys

REPO = os.environ.get("HF_MODEL_REPO", "hosseinzr/neyshekar-whisper-large-v3-lora")

# local path -> path inside the repository. Anything missing is reported and skipped rather
# than aborting, so a partial artefact directory still publishes what it has.
ARTEFACTS = {
    "run_artifacts/error_analysis_zeroshot.csv": "run_artifacts/error_analysis_zeroshot.csv",
    "run_artifacts/error_analysis.csv":          "run_artifacts/error_analysis.csv",
    "run_artifacts/error_categories.csv":        "run_artifacts/error_categories.csv",
    "run_artifacts/train.log":                   "run_artifacts/train.log",
    "run_artifacts/ablation_logs.tar.gz":        "run_artifacts/ablation_logs.tar.gz",
    "figures/fig8_summary.png":                  "figures/fig8_summary.png",
}
CARD = ("hf_model_card.md", "README.md")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--push", action="store_true", help="actually upload (default is a dry run)")
    p.add_argument("--card", action="store_true", help="also replace the model card")
    p.add_argument("--repo", default=REPO)
    args = p.parse_args()

    plan = []
    for local, remote in ARTEFACTS.items():
        if os.path.exists(local):
            plan.append((local, remote, os.path.getsize(local)))
        else:
            print(f"  missing, skipped   {local}")
    if args.card:
        if os.path.exists(CARD[0]):
            plan.append((CARD[0], CARD[1], os.path.getsize(CARD[0])))
        else:
            sys.exit(f"--card given but {CARD[0]} does not exist")

    if not plan:
        sys.exit("nothing to upload")

    print(f"\n  target: {args.repo}\n")
    for local, remote, size in plan:
        print(f"  {size/1024:9,.0f} KB   {local}  ->  {remote}")
    total = sum(s for _, _, s in plan)
    print(f"  {'-'*9}\n  {total/1024:9,.0f} KB   {len(plan)} file(s)\n")

    if not args.push:
        print("  dry run. Re-run with --push to upload.\n")
        return

    # Ask the library where the token is rather than guessing at a cache path: the location has
    # moved between versions, and the CLI itself was renamed from huggingface-cli to hf.
    from huggingface_hub import HfApi, get_token
    if not get_token():
        sys.exit("No credentials found. Run `hf auth login`, or set HF_TOKEN for this session.")
    api = HfApi()
    for local, remote, _ in plan:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=args.repo, repo_type="model",
                        commit_message=f"Add {os.path.basename(remote)}")
        print(f"  uploaded  {remote}")
    print(f"\n  https://huggingface.co/{args.repo}/tree/main\n")


if __name__ == "__main__":
    main()
