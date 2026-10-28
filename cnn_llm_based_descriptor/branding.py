"""Shared project icon; the launcher needs only the Python standard library."""

from pathlib import Path
import tkinter as tk

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def apply_icon(root):
    images = [tk.PhotoImage(master=root, file=str(ASSETS / f"icon-{size}.png"))
              for size in (64, 256)]
    root.iconphoto(True, *images)
    # Keep references alive for Tk and provide a compact in-window mark.
    root.project_icons = images
    root.project_mark = images[0].subsample(2)
    return root.project_mark
