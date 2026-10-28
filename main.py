#!/usr/bin/env python3
"""Start CNN + LLM Based Descriptor. Keep this guard for multiprocessing on all platforms."""

import multiprocessing
import os
from pathlib import Path
import sys


if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = Path(__file__).resolve().parent
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".ultralytics" / "matplotlib"))
    try:
        from cnn_llm_based_descriptor.app import main
        main()
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Open ./Run_CNN_LLM_Based_Descriptor.sh and use Set up / repair.", file=sys.stderr)
        raise SystemExit(1)
