"""
Root launcher script for dataset module.
Re-exports dataset utilities from src/dataset.py
"""
import sys
import os

from src.dataset import (
    DataCollatorSpeechSeq2SeqWithPadding,
    prepare_dataset,
    load_custom_dataset,
    get_datasets_and_collator
)

if __name__ == '__main__':
    print("Testing root dataset.py module launcher...")
