"""Make the package importable as `nodes` however pytest is invoked.

The repo is a ComfyUI custom-node folder, not an installed package, so there is no
entry in sys.path pointing at it. conftest.py is imported by pytest before any test
module, which makes it the one place this has to be done — previously every test file
repeated the same sys.path insert, and that only worked when the file was run directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
