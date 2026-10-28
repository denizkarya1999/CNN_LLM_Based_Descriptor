#!/usr/bin/env bash
# Compatibility entry point for existing shortcuts.
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/Run_YOLO_LLM_Visual_Descriptor.sh" "$@"
