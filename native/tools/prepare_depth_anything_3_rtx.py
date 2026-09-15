"""Setup-only depth-anything-3 TensorRT preparation."""
import sys
from pathlib import Path

# The installer uses Python isolated mode, which omits the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_tensorrt_rtx import main

if __name__ == "__main__":
    main("depth-anything-3")
