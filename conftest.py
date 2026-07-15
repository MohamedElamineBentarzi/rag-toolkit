"""Pytest picks this up at the repo root and prepends this directory to
sys.path, so `import rag_blocks` works on a fresh clone before any
`pip install -e .`. Standard trick for flat-layout projects."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
