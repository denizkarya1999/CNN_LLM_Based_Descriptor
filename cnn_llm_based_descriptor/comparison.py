"""Compare independent descriptor methods on one frozen detector result."""
from copy import deepcopy
import csv
from dataclasses import replace
import json
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .methods import SEARCH_ALGORITHMS, method_details, NO_ENVIRONMENT_BACKENDS
from .measurement import measure_description, efficiency_text
from .scenes import SETTINGS
from .theme import BG, PANEL, FG, MUTED
from .window import resizable_window, FlowRow, wrap_label

UNLABELED = 'Not labeled'


def comparison_configs(cfg):
    return ([replace(cfg, describe_enabled=True, describe_backend='Search Algorithm', describe_search=algorithm)
             for algorithm in SEARCH_ALGORITHMS] +
            [replace(cfg, describe_enabled=True, describe_backend='Probability Algorithm'),
             replace(cfg, describe_enabled=True, describe_backend='General Algorithmic Approach'),
             replace(cfg, describe_enabled=True, describe_backend='LLM')])


def expected_matches(result, decision=UNLABELED, setting=UNLABELED):
    valid = result.get('grounding_status') == 'validated' and not result.get('error')
    # Invalid answers are incorrect when a reference exists, never a successful No Action.
    return {
        'decision_match': None if decision == UNLABELED else bool(valid and result.get('action_decision') == decision),
        'setting_match': (None if setting == UNLABELED or result['descriptor_type'] in NO_ENVIRONMENT_BACKENDS
                          else bool(valid and result.get('setting') == setting)),
    }


def run_comparison(cfg, frame, records, repeats, emit, cancelled):
    """Same pixels/YOLO result, serial methods, no new detector inference.

    Method order rotates between repetitions. Timing is descriptive: loading,
    OS/GPU contention and input differences are reported, not normalized away.
    """
    from .descriptor import create_descriptor_client
    configs = comparison_configs(cfg)
    clients = [create_descriptor_client(c) for c in configs]
    for repetition in range(repeats):
        order = list(range(len(configs)))
        order = order[repetition % len(order):] + order[:repetition % len(order)]
        for index in order:
            if cancelled.is_set():
                return
            current = configs[index]
            emit('progress', f"Run {repetition+1}/{repeats} · {method_details(current)['algorithm']}")
            started = time.perf_counter_ns()
            try:
                current.validate_descriptor()
                result = measure_description(clients[index], frame.copy(), deepcopy(records))
            except Exception as exc:
                result = {**method_details(current), 'error': str(exc), 'grounding_status': 'error',
                          'inference_latency_ms': (time.perf_counter_ns()-started)/1_000_000,
                          'text': str(exc)}
            if cancelled.is_set():
                return
            result.update(repetition=repetition+1, total_input_predictions=len(records),
                          input_pixels=current.describe_backend == 'LLM',
                          input_note=('Image + detector evidence; LLM object/context limit applies.' if current.describe_backend == 'LLM'
                                      else 'All eligible detector labels, scores and boxes; no image pixels.'))
            emit('result', result)


def export_results(path, results, decision, setting, context):
    records = [{**row, **expected_matches(row, decision, setting),
                'sample_id': context.get('sample_id'), 'frame': context.get('frame'),
                'source_seconds': context.get('source_seconds'),
                'expected_decision': None if decision == UNLABELED else decision,
                'expected_setting': None if setting == UNLABELED else setting} for row in results]
    if str(path).lower().endswith('.json'):
        from pathlib import Path
        Path(path).write_text(json.dumps({'context': context, 'results': records}, indent=2), encoding='utf-8')
        return
    fields = ['sample_id', 'frame', 'source_seconds', 'descriptor_type', 'algorithm', 'repetition', 'inference_latency_ms', 'grounding_status',
              'action_decision', 'action_plan_name', 'setting', 'expected_decision', 'decision_match', 'expected_setting',
              'setting_match', 'detections_considered', 'total_input_predictions', 'input_pixels',
              'input_note', 'efficiency', 'error']
    with open(path, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(row[key]) if key == 'efficiency' and key in row else row.get(key, '') for key in fields})


class ComparisonWindow:
    def __init__(self, parent, cfg, frame, records, context):
        self.cfg, self.frame, self.records, self.context = cfg, frame.copy(), deepcopy(records), context
        self.results, self.messages = [], queue.Queue()
        self.cancelled = threading.Event()
        self.running = self.closed = False
        self.window = window = tk.Toplevel(parent)
        window.title('Compare LLM and conventional AI')
        window.configure(bg=BG); resizable_window(window, 1180, 820, minimum=(760, 520))
        self.decision = tk.StringVar(value=UNLABELED)
        self.setting = tk.StringVar(value=UNLABELED)
        self.repeats = tk.StringVar(value='1')
        self.status = tk.StringVar(value=f"Frozen frame {context.get('frame', '—')} · {len(records)} YOLO predictions")
        footer = ttk.Frame(window, padding=12); footer.pack(side='bottom', fill='x')
        buttons = FlowRow(footer); buttons.pack(fill='x')
        self.run_button = buttons.add(ttk.Button(buttons, text='Run comparison', command=self.run, style='Start.TButton'))
        self.cancel_button = buttons.add(ttk.Button(buttons, text='Cancel', command=self.cancel, state='disabled'))
        self.export_button = buttons.add(ttk.Button(buttons, text='Export CSV / JSON', command=self.export, state='disabled'))
        buttons.add(ttk.Button(buttons, text='Close', command=self.close))
        wrap_label(ttk.Label(footer, textvariable=self.status, foreground=MUTED)).pack(fill='x', pady=(6, 0))
        outer = ttk.Frame(window, padding=12); outer.pack(fill='both', expand=True)
        wrap_label(ttk.Label(outer, text='Same frozen frame and detector result · latency, reference-label matches, and method-specific work', font=('DejaVu Sans', 11, 'bold'))).pack(fill='x')
        wrap_label(ttk.Label(outer, text='LLM also sees the image and may retain fewer detections. Search and robot plans have no environment output. Loading is included; work counts use different units and are not energy or memory measurements.', foreground=MUTED)).pack(fill='x', pady=6)
        controls = FlowRow(outer); controls.pack(fill='x')
        for label, var, values, width in (
                ('Expected decision', self.decision, (UNLABELED, 'No Action', 'Take Action'), 15),
                ('Expected setting', self.setting, (UNLABELED, *SETTINGS), 18)):
            group = ttk.Frame(controls)
            ttk.Label(group, text=label).pack(anchor='w')
            combo = ttk.Combobox(group, textvariable=var, values=values, state='readonly', width=width)
            combo.pack(fill='x'); combo.bind('<<ComboboxSelected>>', lambda _e: self.refresh())
            controls.add(group)
        group = ttk.Frame(controls); ttk.Label(group, text='Repeated runs').pack(anchor='w')
        self.repeat_control = ttk.Spinbox(group, from_=1, to=5, textvariable=self.repeats, width=5)
        self.repeat_control.pack(); controls.add(group)
        wrap_label(ttk.Label(outer, text='Accuracy needs independent reference labels. Match/mismatch below applies only to this frame; repeated runs are not new accuracy samples.', foreground=MUTED)).pack(fill='x', pady=6)
        results_area = ttk.Frame(outer); results_area.pack(fill='both', expand=True)
        results_area.columnconfigure(0, weight=1)
        results_area.rowconfigure(0, weight=3, minsize=65)
        results_area.rowconfigure(1, weight=2, minsize=45)
        table = ttk.Frame(results_area); table.grid(row=0, column=0, sticky='nsew')
        columns = ('method', 'run', 'latency', 'decision', 'setting', 'decision_match', 'setting_match', 'coverage', 'work')
        style = ttk.Style(window)
        style.configure('Comparison.Treeview', background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=26)
        style.configure('Comparison.Treeview.Heading', background=BG, foreground=FG)
        self.table = ttk.Treeview(table, columns=columns, show='headings', height=3, style='Comparison.Treeview')
        names = ('Method', 'Run', 'Latency (ms)', 'Decision', 'Setting', 'Decision match', 'Setting match', 'Evidence', 'Efficiency / work')
        widths = (285, 42, 100, 100, 85, 115, 125, 80, 300)
        for key, name, width in zip(columns, names, widths):
            self.table.heading(key, text=name); self.table.column(key, width=width, minwidth=width, stretch=(key == 'work'))
        self.table.grid(row=0, column=0, sticky='nsew')
        vs = ttk.Scrollbar(table, orient='vertical', command=self.table.yview); vs.grid(row=0, column=1, sticky='ns')
        hs = ttk.Scrollbar(table, orient='horizontal', command=self.table.xview); hs.grid(row=1, column=0, sticky='ew')
        self.table.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        table.columnconfigure(0, weight=1); table.rowconfigure(0, weight=1)
        self.table.bind('<<TreeviewSelect>>', self.select)
        self.table.tag_configure('take_action', foreground='#ff6b78')
        self.table.tag_configure('no_action', foreground='#65e08c')
        detail = ttk.Frame(results_area); detail.grid(row=1, column=0, sticky='nsew', pady=(8, 0))
        self.detail = tk.Text(detail, height=3, width=1, wrap='word', font=('DejaVu Sans', 10), bg=PANEL, fg=FG, state='disabled', relief='flat')
        self.detail.pack(side='left', fill='both', expand=True)
        scroll = ttk.Scrollbar(detail, command=self.detail.yview); scroll.pack(side='right', fill='y')
        self.detail.configure(yscrollcommand=scroll.set)
        window.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = window.after(80, self.poll)

    def run(self):
        if self.running:
            return
        try:
            repeats = int(self.repeats.get())
            if not 1 <= repeats <= 5:
                raise ValueError
        except ValueError:
            messagebox.showerror('Repeated runs', 'Choose 1 to 5 runs.', parent=self.window); return
        self.results.clear(); self.refresh(); self.cancelled.clear(); self.running = True
        self.run_button.configure(state='disabled'); self.cancel_button.configure(state='normal')
        self.export_button.configure(state='disabled'); self.repeat_control.configure(state='disabled')
        def work():
            try:
                run_comparison(self.cfg, self.frame, self.records, repeats,
                               lambda kind, value: self.messages.put((kind, value)), self.cancelled)
            except Exception as exc:
                self.messages.put(('progress', str(exc)))
            finally:
                self.running = False
                self.messages.put(('done', None))
        threading.Thread(target=work, daemon=True).start()

    def cancel(self):
        self.cancelled.set(); self.status.set('Cancelling · the current local request may finish before it releases the model.')

    def refresh(self):
        selection = self.table.selection()
        self.table.delete(*self.table.get_children())
        for i, result in enumerate(self.results):
            matches = expected_matches(result, self.decision.get(), self.setting.get())
            def label(value): return 'Not scored' if value is None else 'Match' if value else 'Mismatch'
            setting_match = 'Not applicable' if result['descriptor_type'] in NO_ENVIRONMENT_BACKENDS else label(matches['setting_match'])
            decision = result.get('action_decision', 'Error') if result.get('grounding_status') == 'validated' else 'Unavailable'
            values = (result['algorithm'], result['repetition'], f"{result['inference_latency_ms']:.3f}",
                      decision, result.get('setting') or '—', label(matches['decision_match']), setting_match,
                      f"{result.get('detections_considered', len(result.get('evidence', [])))}/{result['total_input_predictions']}", efficiency_text(result))
            self.table.insert('', 'end', iid=str(i), values=values,
                              tags=('take_action' if decision == 'Take Action' else 'no_action' if decision == 'No Action' else '',))
        if selection and self.table.exists(selection[0]):
            self.table.selection_set(selection)
        elif self.results:
            self.table.selection_set(str(len(self.results)-1))
        self.select()

    def select(self, _event=None):
        selected = self.table.selection()
        if selected:
            result = self.results[int(selected[0])]
            text = (f"{result['descriptor_type']} · {result['algorithm']}\n"
                    f"Latency: {result['inference_latency_ms']:.3f} ms\n"
                    f"Efficiency: {efficiency_text(result)}\n\n" + result['text'])
        else:
            text = 'Select a result to read its explanation, traversal, calculations, or robot plan.'
        self.detail.configure(state='normal'); self.detail.delete('1.0', 'end'); self.detail.insert('1.0', text); self.detail.configure(state='disabled')

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == 'progress': self.status.set(value)
                elif kind == 'result': self.results.append(value); self.refresh()
                elif kind == 'done':
                    self.running = False
                    self.run_button.configure(state='normal'); self.cancel_button.configure(state='disabled')
                    self.export_button.configure(state='normal' if self.results else 'disabled'); self.repeat_control.configure(state='normal')
                    self.status.set(('Cancelled' if self.cancelled.is_set() else 'Complete') + f' · {len(self.results)} results · add reference labels to score matches; export for dataset evaluation.')
        except queue.Empty:
            pass
        self.timer = self.window.after(80, self.poll)

    def export(self):
        path = filedialog.asksaveasfilename(parent=self.window, title='Export comparison', defaultextension='.csv',
                                          filetypes=[('CSV table', '*.csv'), ('Full results and reasoning', '*.json')])
        if path:
            try:
                export_results(path, self.results, self.decision.get(), self.setting.get(), self.context)
                self.status.set('Comparison exported.')
            except OSError as exc:
                messagebox.showerror('Could not export', str(exc), parent=self.window)

    def close(self):
        self.cancelled.set(); self.closed = True
        self.window.after_cancel(self.timer); self.window.destroy()
