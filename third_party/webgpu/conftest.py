"""
Pytest configuration for WebGPU backend tests.

Ensures the `backend` package can be imported when running
tests from the repository root:

    python -m pytest third_party/webgpu/ -v
"""

import sys
import os

# Add the webgpu directory to sys.path so `from backend.xxx import ...` works.
sys.path.insert(0, os.path.dirname(__file__))
