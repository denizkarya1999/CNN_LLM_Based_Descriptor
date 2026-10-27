"""Resizable windows and rows that wrap controls without clipping."""
from tkinter import ttk


def resizable_window(window, width, height, minimum=(760, 520)):
    width = min(width, window.winfo_screenwidth() - 40)
    height = min(height, window.winfo_screenheight() - 70)
    window.resizable(True, True)
    window.minsize(min(minimum[0], width), min(minimum[1], height))
    # No maximum: leave maximize and multi-monitor resizing to the desktop.
    window.geometry(f'{width}x{height}')
    return width, height


def wrap_label(label, padding=0):
    """Wrap text to its allocated width, including long file names and errors."""
    label.configure(width=1)
    def resized(event):
        length = max(40, event.width - padding)
        if int(label.cget('wraplength') or 0) != length:
            label.configure(wraplength=length)
    label.bind('<Configure>', resized, add='+')
    return label


class FlowRow(ttk.Frame):
    """Place whole controls on additional rows as space becomes limited."""
    def __init__(self, parent, gap=8, **kwargs):
        super().__init__(parent, **kwargs)
        self.gap = gap
        self.items = []
        self.job = None
        self.positions = {}
        self.bind('<Configure>', lambda _e: self.reflow())
        self.bind('<Destroy>', self._destroy)

    def add(self, widget, expand=False):
        self.items.append((widget, expand))
        widget.bind('<Configure>', lambda _e: self.reflow(), add='+')
        self.reflow()
        return widget

    def reflow(self):
        if self.job is None and self.winfo_exists():
            self.job = self.after_idle(self._layout)

    def _layout(self):
        self.job = None
        available = max(1, self.winfo_width())
        rows, row, used = [], [], 0
        for widget, expand in self.items:
            width = min(available, widget.winfo_reqwidth())
            height = widget.winfo_reqheight()
            if row and used + self.gap + width > available:
                rows.append(row); row = []; used = 0
            if row: used += self.gap
            row.append((widget, expand, width, height)); used += width
        if row: rows.append(row)
        y = 0
        for row in rows:
            extra = max(0, available - sum(item[2] for item in row) - self.gap * (len(row)-1))
            growing = sum(item[1] for item in row)
            x, height = 0, max(item[3] for item in row)
            for widget, expand, width, h in row:
                if expand and growing:
                    addition = extra // growing
                    width += addition; extra -= addition; growing -= 1
                position = (x, y + (height-h)//2, width, h)
                if self.positions.get(widget) != position:
                    widget.place(x=position[0], y=position[1], width=width, height=h)
                    self.positions[widget] = position
                x += width + self.gap
            y += height + self.gap
        wanted = max(1, y - self.gap)
        if self.winfo_reqheight() != wanted:
            self.configure(height=wanted)

    def _destroy(self, event):
        if event.widget is self and self.job is not None:
            self.after_cancel(self.job)
            self.job = None
