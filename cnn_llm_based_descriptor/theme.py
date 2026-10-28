"""Shared desktop colors and controls; usable before optional packages install."""
from tkinter import ttk

BG = '#0d1420'
PANEL = '#172334'
FG = '#edf3fa'
MUTED = '#a4b5ca'
ACCENT = '#69dfc0'
AMBER = '#ffd084'


def apply_theme(root, body_font=('DejaVu Sans', 10), bold_font=('DejaVu Sans', 10, 'bold')):
    root.configure(bg=BG)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=BG, foreground=FG, font=body_font)
    style.configure('Card.TFrame', background=PANEL)
    style.configure('Card.TLabel', background=PANEL)
    style.configure('Muted.TLabel', foreground=MUTED)
    style.configure('TButton', background='#26374d', foreground=FG, padding=(12, 8), borderwidth=0)
    style.map('TButton', background=[('active', '#344d68'), ('disabled', '#1c2a3b')],
              foreground=[('disabled', '#728399')])
    for name in ('Start.TButton', 'Open.TButton'):
        style.configure(name, background=ACCENT, foreground='#082c26', font=bold_font)
        style.map(name, background=[('active', '#8cebd4'), ('disabled', '#26374d')],
                  foreground=[('disabled', '#728399')])
    style.configure('Record.TButton', background='#643440', foreground='#ffdfe4')
    style.configure('Horizontal.TProgressbar', background=ACCENT, troughcolor=PANEL, borderwidth=0, thickness=6)
    style.configure('TEntry', fieldbackground=PANEL, insertcolor=FG, padding=6, bordercolor='#34475d')
    style.configure('TCombobox', fieldbackground=PANEL, padding=6, arrowsize=14)
    style.map('TCombobox', fieldbackground=[('readonly', PANEL)], foreground=[('readonly', FG), ('disabled', '#728399')])
    style.configure('TCheckbutton', padding=(0, 6))
    style.configure('TNotebook', borderwidth=0)
    style.configure('TNotebook.Tab', background=PANEL, padding=(16, 9))
    style.map('TNotebook.Tab', background=[('selected', '#263b50')], foreground=[('selected', ACCENT)])
    style.configure('Vertical.TScrollbar', background='#34475d', troughcolor=PANEL, borderwidth=0, arrowsize=12)
    root.option_add('*TCombobox*Listbox.background', PANEL)
    root.option_add('*TCombobox*Listbox.foreground', FG)
    root.option_add('*TCombobox*Listbox.font', body_font)
