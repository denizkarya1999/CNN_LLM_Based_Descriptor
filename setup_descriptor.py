#!/usr/bin/env python3
"""Download and verify the default local quantized vision model."""

import shutil
import threading
import sys

from cnn_llm_based_descriptor.algorithms.llm.client import DEFAULT_MODEL
from cnn_llm_based_descriptor.algorithms.llm.transport import request_json, download_model
from cnn_llm_based_descriptor.algorithms.llm.model_store import PROJECT_ENDPOINT, MODEL_DIRECTORY


def main():
    if not shutil.which("ollama"):
        raise RuntimeError("Install Ollama from https://ollama.com/download, then run this script again.")
    print(f"Downloading {DEFAULT_MODEL} (Q4_K_M, about 1.9 GB) into {MODEL_DIRECTORY}.", flush=True)
    download_model(PROJECT_ENDPOINT, DEFAULT_MODEL, lambda message: print(message, flush=True), threading.Event())
    model = request_json(PROJECT_ENDPOINT, "/api/show", {"model": DEFAULT_MODEL}, 15)
    quantization = model.get("details", {}).get("quantization_level", "")
    if "vision" not in model.get("capabilities", []) or quantization != "Q4_K_M":
        raise RuntimeError("The installed model does not report vision support with Q4_K_M quantization. Update Ollama and retry.")
    print(f"Ready: {DEFAULT_MODEL}, {quantization}. Open Run_CNN_LLM_Based_Descriptor.sh.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print(f"Vision setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
