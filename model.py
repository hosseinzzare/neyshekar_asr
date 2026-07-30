"""
Root model launcher script.
Re-exports get_whisper_qlora_model from src/model.py
"""
import sys
import os

from src.model import get_whisper_qlora_model, make_inputs_require_grad

if __name__ == '__main__':
    print("Testing root model.py launcher...")
