"""
Launcher script for data preparation.
Imports and runs src.data_prep
"""
import sys
import os

from src.data_prep import prepare_and_save_data

if __name__ == '__main__':
    prepare_and_save_data()
