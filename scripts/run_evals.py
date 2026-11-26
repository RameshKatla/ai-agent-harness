"""Run the agent regression suite from a source checkout."""

import sys
from pathlib import Path

# Executing a script directly adds its own directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
