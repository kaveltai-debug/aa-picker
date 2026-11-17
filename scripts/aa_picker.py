"""CLI wrapper for aa_picker package."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from aa_picker.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
