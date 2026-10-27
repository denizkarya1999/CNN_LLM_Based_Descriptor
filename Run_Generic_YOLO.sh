#!/usr/bin/env bash
# Single entry point. All setup choices are made in the graphical launcher.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

show_error() {
  if command -v zenity >/dev/null; then
    zenity --error --title='Generic YOLO' --text="$1" --width=460
  elif command -v xmessage >/dev/null; then
    xmessage -center "$1"
  else
    printf '%s\n' "$1" >&2
  fi
}

if ! command -v python3 >/dev/null || ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  show_error 'Generic YOLO needs Python 3.10 or later. Install Python through your operating system software manager, then open this launcher again.'
  exit 1
fi
if ! python3 -c 'import tkinter' 2>/dev/null; then
  if command -v zenity >/dev/null && command -v pkexec >/dev/null && command -v apt-get >/dev/null; then
    if zenity --question --title='Generic YOLO setup' --width=460 \
      --text='Python GUI support is missing. Install it now? Your system will ask for administrator authentication.'; then
      if ! pkexec "$(command -v apt-get)" install -y python3-tk python3-venv python3-pip; then
        show_error 'GUI support could not be installed. Check your system package manager and try this launcher again.'
        exit 1
      fi
    else
      exit 0
    fi
  else
    show_error 'Python Tk GUI support is missing. Install the python3-tk package through your operating system software manager, then open this launcher again.'
    exit 1
  fi
fi
exec python3 launcher.py
