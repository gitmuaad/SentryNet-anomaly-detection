"""Make ``src/`` importable when scripts are run directly (no install step required).

Students can run ``python scripts/<name>.py`` from the repository root without setting
``PYTHONPATH`` or installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
