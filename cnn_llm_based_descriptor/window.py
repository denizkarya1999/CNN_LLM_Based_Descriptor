"""Resizable windows and rows that wrap controls without clipping."""
import tkinter as tk
from tkinter import ttk


class ScrollFrame(ttk.Frame):
    """Scrollable form with a visible scrollbar and keyboard focus tracking."""
    def __init__(self, parent, padding=0, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background='#0d1420', width=1, height=1)
        self.bar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.bar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.body = ttk.Frame(self.canvas, padding=padding)
        self.window = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.bind('<Configure>', self._resize)
        self.body.bind('<Configure>', lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.top = self.winfo_toplevel()
        self.bindings = [(event, self.top.bind(event, callback, add='+')) for event, callback in
                         [('<MouseWheel>', self._wheel), ('<Button-4>', self._wheel),
                          ('<Button-5>', self._wheel), ('<FocusIn>', self._focus)]]
        self.bind('<Destroy>', self._destroy)

    def _contains(self, widget):
        return str(widget).startswith(str(self.body) + '.') or widget in (self.body, self.canvas)

    def _resize(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def _wheel(self, event):
        if not self._contains(event.widget) or isinstance(event.widget, (tk.Text, ttk.Combobox)):
            return
        if self.body.winfo_height() > self.canvas.winfo_height():
            direction = -1 if getattr(event, 'num', 0) == 4 or getattr(event, 'delta', 0) > 0 else 1
            self.canvas.yview_scroll(direction * 3, 'units')
            return 'break'

    def _focus(self, event):
        if not self._contains(event.widget):
            return
        top = event.widget.winfo_rooty() - self.canvas.winfo_rooty()
        bottom = top + event.widget.winfo_height()
        offset = top if top < 0 else bottom - self.canvas.winfo_height() if bottom > self.canvas.winfo_height() else 0
        if offset:
            self.canvas.yview_moveto(self.canvas.yview()[0] + offset / max(1, self.body.winfo_height()))

    def _destroy(self, event):
        if event.widget is self:
            for sequence, identifier in self.bindings:
                self.top.unbind(sequence, identifier)


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
