"""
Force Latin digits in an existing .docx.

Same fix docbuild.py applies at build time, for the one document whose build script is no
longer in the tree. Idempotent: re-running it changes nothing.

    python fix_digits.py docs/Task1_Dataset_Investigation_detailed.docx
"""
import sys
from docx import Document
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from docbuild import _tag_run_languages, DOCS
import os

for name in sys.argv[1:]:
    path = name if os.path.isabs(name) else os.path.join(DOCS, os.path.basename(name))
    d = Document(path)
    _tag_run_languages(d)
    d.save(path)
    print("fixed:", os.path.basename(path))
