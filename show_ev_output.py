from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PYTHON_DIR = ROOT / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from ev_runner import main


if __name__ == "__main__":
    main()
