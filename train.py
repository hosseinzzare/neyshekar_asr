"""
Root launcher script for Whisper Fine-Tuning.
Invokes src/train.py
"""
import sys
import os

from src.train import run_training_pipeline

if __name__ == '__main__':
    run_training_pipeline()
