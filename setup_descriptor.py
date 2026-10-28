#!/usr/bin/env python3
"""Download and verify the default local quantized vision model."""

import shutil
import subprocess
import sys

from yolo_llm_visual_descriptor.descriptor import DEFAULT_MODEL, request_json


def main():
    if not shutil.which("ollama"):
        raise RuntimeError("Install Ollama from https://ollama.com/download, start it, and run this script again.")
    print(f"Downloading {DEFAULT_MODEL} (Q4_0, about 1.7 GB) into Ollama's local cache.", flush=True)
    subprocess.run(["ollama", "pull", DEFAULT_MODEL], check=True)
    model = request_json("http://127.0.0.1:11434", "/api/show", {"model": DEFAULT_MODEL}, 15)
    quantization = model.get("details", {}).get("quantization_level", "")
    if "vision" not in model.get("capabilities", []) or quantization != "Q4_0":
        raise RuntimeError("The installed model does not report vision support with Q4_0 quantization. Update Ollama and retry.")
    print(f"Ready: {DEFAULT_MODEL}, {quantization}. Open Run_YOLO_LLM_Visual_Descriptor.sh.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Vision setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
