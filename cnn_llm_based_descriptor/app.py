"""Compact camera-first desktop GUI; advanced controls live in Settings."""
from collections import deque
from dataclasses import asdict, fields, replace
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
import time
import threading
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font as tkfont
from PIL import Image, ImageTk
from .config import ROOT, RunConfig, SCALES, TASKS, RECORD_MODES
from .engine import infer, probe_devices
from .descriptor import download_model
from .recording import recording_destinations
from .cameras import discover_cameras, linux_camera_path
from .window import resizable_window, FlowRow, ScrollFrame, wrap_label
from .theme import apply_theme, BG, PANEL, FG, MUTED, ACCENT, AMBER
from .model_store import PROJECT_ENDPOINT, MODEL_DIRECTORY
from .branding import apply_icon
from .config import DESCRIBE_BACKENDS



class App:
    def __init__(self, root, worker_target=infer):
        self.root = root
        self.worker_target = worker_target
        self.ctx = mp.get_context('spawn')
        self.worker = self.probe = None
        self.messages = self.previews = self.commands = self.probe_queue = None
        self.stop_event = self.pause_event = self.inference_switch = None
        self.stopping_since = None
        self.closing = self.ready = self.recording = self.record_pending = False
        self.outcome = self.last_output = None
        self.last_error = None
        self.image = self.photo = None
        self.resize_job = None
        self.layout_job = None
        self.layout_size = None
        self.font_size = 10
        self.settings_window = self.details_window = None
        self.settings_path = ROOT / 'settings.json'
        self.log_lines = deque(maxlen=500)
        self.latest_records = []
        self.last_packet = None
        self.latest_description = None
        self.system_text = ''
        self.device_map = {'CPU': 'cpu'}
        self.camera_map = {}
        self.named_cameras = sys.platform.startswith('linux')
        self.selected_camera = tk.StringVar(value='Finding cameras…')
        initial = self._load_settings()
        self.vars = {f.name: (tk.BooleanVar if isinstance(getattr(initial, f.name), bool) else tk.StringVar)(value=getattr(initial, f.name)) for f in fields(RunConfig)}
        self.selected_device = tk.StringVar(value='Checking devices…')
        self.status = tk.StringVar(value='Checking available devices…')
        self.stats = tk.StringVar(value='Open a camera to begin')
        self.record_status = tk.StringVar(value='Not recording')
        self.model_messages = queue.Queue(maxsize=8)
        self.model_cancel = threading.Event()
        self.model_job = None
        self.model_status = tk.StringVar(value='Qwen3-VL 2B · 4-bit · about 1.9 GB · local Ollama')
        self.description_title = tk.StringVar(value='Visual description · waiting for a frame')
        self.description_text = tk.StringVar(value='Enable descriptions in Settings for Detections, Environment, Take Action or No Action, and Reason.')
        self._style()
        self._build()
        root.bind('<Configure>', self._window_resized, add='+')
        self._source_changed(reset=False)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Escape>', lambda _e: self.stop())
        root.bind('<Control-Return>', lambda _e: self.start())
        self.timer = root.after(40, self._pump)
        self.refresh_devices()

    def _load_settings(self):
        try:
            data = json.loads(self.settings_path.read_text())
            if data.get('describe_url') in ('http://127.0.0.1:11434', 'http://localhost:11434'):
                data['describe_url'] = PROJECT_ENDPOINT
            cfg = RunConfig(**{f.name: data[f.name] for f in fields(RunConfig) if f.name in data})
            if cfg.source_kind in {'Video', 'Image'} and not Path(cfg.source).expanduser().is_file():
                cfg = replace(cfg, source_kind='Webcam', source='0')
            return cfg
        except (OSError, ValueError, TypeError):
            if platform.machine().lower() in {'arm64', 'aarch64'}:
                return RunConfig(device='cpu', half=False, image_size=320, max_detections=50, frame_stride=2)
            return RunConfig()

    def _style(self):
        self.root.title('CNN + LLM Based Descriptor')
        self.brand_icon = apply_icon(self.root)
        self.window_width, _ = resizable_window(self.root, 1440, 900)
        self.body_font = tkfont.Font(root=self.root, family='DejaVu Sans', size=10)
        self.bold_font = tkfont.Font(root=self.root, family='DejaVu Sans', size=10, weight='bold')
        self.title_font = tkfont.Font(root=self.root, family='DejaVu Sans', size=20, weight='bold')
        apply_theme(self.root, self.body_font, self.bold_font)

    def _build(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill='both', expand=True)
        # Pack the footer first: even a long description or a small display
        # cannot push the session controls below the window.
        footer = ttk.Frame(outer)
        footer.pack(side='bottom', fill='x', pady=(12, 0))
        ttk.Separator(footer).pack(fill='x', pady=(0, 10))
        actions = FlowRow(footer); actions.pack(fill='x')
        self.start_button = actions.add(ttk.Button(actions, text='Open camera', style='Start.TButton', command=self.start, width=10))
        self.stop_button = actions.add(ttk.Button(actions, text='Stop', command=self.stop, state='disabled', width=5))
        self.record_button = actions.add(ttk.Button(actions, text='● Record', style='Record.TButton', command=self.toggle_record, state='disabled', width=13))
        self.snapshot_button = actions.add(ttk.Button(actions, text='Snapshot', command=self.snapshot, state='disabled', width=9))
        mode_group = ttk.Frame(actions)
        ttk.Label(mode_group, text='Save').pack(side='left', padx=(4, 6))
        self.mode_combo = ttk.Combobox(mode_group, textvariable=self.vars['record_mode'], values=RECORD_MODES, state='readonly', width=13)
        self.mode_combo.pack(side='left', fill='x', expand=True)
        actions.add(mode_group, expand=True)
        ttk.Label(footer, textvariable=self.status, foreground=MUTED, width=1, anchor='w').pack(fill='x', pady=(8, 0))

        header = ttk.Frame(outer); header.pack(fill='x', pady=(0, 14))
        header.columnconfigure(0, weight=1)
        self.heading_label = wrap_label(ttk.Label(header, text='CNN + LLM Based Descriptor', image=self.brand_icon, compound='left', font=self.title_font))
        self.heading_label.grid(row=0, column=0, sticky='ew', padx=(0, 20))
        ttk.Label(header, text='LOCAL VISION  /  DETECT · DESCRIBE · DECIDE', foreground=MUTED, font=('DejaVu Sans', 9)).grid(row=1, column=0, sticky='w', pady=(4, 0))
        self.tools_row = tools = FlowRow(header, width=345); tools.grid(row=0, column=1, rowspan=2, sticky='ew')
        self.settings_button = tools.add(ttk.Button(tools, text='Settings', command=self.show_settings, width=7))
        self.details_button = tools.add(ttk.Button(tools, text='Details', command=self.show_details, width=7))
        self.outputs_button = tools.add(ttk.Button(tools, text='Outputs', command=self.open_outputs, width=7))

        source_card = ttk.Frame(outer, padding=12, style='Card.TFrame')
        source_card.pack(fill='x', pady=(0, 12))
        inputs = ttk.Frame(source_card, style='Card.TFrame'); inputs.pack(fill='x')
        inputs.columnconfigure(1, weight=1)
        ttk.Label(inputs, text='SOURCE', style='Card.TLabel', foreground=MUTED, font=('DejaVu Sans', 9, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 5))
        self.source_combo = ttk.Combobox(inputs, textvariable=self.vars['source_kind'], values=['Webcam','Pi camera (CSI)','Network stream','Video','Image'], state='readonly', width=16)
        self.source_combo.grid(row=1, column=0, sticky='ew', padx=(0, 10))
        self.source_combo.bind('<<ComboboxSelected>>', lambda _e: self._source_changed())
        self.source_label = ttk.Label(inputs, text='Camera', style='Card.TLabel', foreground=MUTED)
        self.source_label.grid(row=0, column=1, sticky='w', pady=(0, 5))
        self.source_entry = ttk.Entry(inputs, textvariable=self.vars['source'])
        self.source_entry.grid(row=1, column=1, sticky='ew')
        self.camera_combo = ttk.Combobox(inputs, textvariable=self.selected_camera, state='readonly')
        self.camera_combo.grid(row=1, column=1, sticky='ew')
        self.camera_combo.bind('<<ComboboxSelected>>', lambda _e: self._camera_selected())
        self.refresh_button = ttk.Button(inputs, text='Refresh', command=self.refresh_cameras)
        self.refresh_button.grid(row=1, column=2, padx=(8, 0))
        self.browse_button = ttk.Button(inputs, text='Choose file…', command=self.browse_source)
        self.browse_button.grid(row=1, column=2, padx=(8, 0))

        model = FlowRow(outer); model.pack(fill='x', pady=(0, 12))
        self.inference_button = model.add(ttk.Checkbutton(model, text='YOLO inference', variable=self.vars['inference_enabled'], command=self.inference_changed))
        size_group = ttk.Frame(model)
        ttk.Label(size_group, text='Model').pack(side='left', padx=(0, 6))
        self.scale_combo = ttk.Combobox(size_group, textvariable=self.vars['scale'], values=list(SCALES), state='readonly', width=14)
        self.scale_combo.pack(side='left'); model.add(size_group)
        device_group = ttk.Frame(model)
        ttk.Label(device_group, text='Device').pack(side='left', padx=(8, 6))
        self.device_combo = ttk.Combobox(device_group, textvariable=self.selected_device, values=list(self.device_map), state='readonly', width=22)
        self.device_combo.pack(side='left', fill='x', expand=True)
        model.add(device_group, expand=True)

        self.workspace = ttk.Frame(outer); self.workspace.pack(fill='both', expand=True)
        self.workspace_mode = None
        self.preview_card = ttk.Frame(self.workspace, style='Card.TFrame', padding=10)
        self.preview_card.rowconfigure(1, weight=1); self.preview_card.columnconfigure(0, weight=1)
        preview_header = ttk.Frame(self.preview_card, style='Card.TFrame')
        preview_header.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        ttk.Label(preview_header, text='Live preview', style='Card.TLabel', font=self.bold_font).pack(side='left')
        self.record_label = ttk.Label(preview_header, textvariable=self.record_status, style='Card.TLabel', foreground=MUTED)
        self.record_label.pack(side='right')
        self.canvas = tk.Canvas(self.preview_card, background='#080e17', highlightthickness=0, width=1, height=1)
        self.canvas.grid(row=1, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', self._resize)
        ttk.Label(self.preview_card, textvariable=self.stats, style='Card.TLabel', foreground=MUTED, width=1).grid(row=2, column=0, sticky='ew', pady=(8, 0))

        self.description_card = ttk.Frame(self.workspace, style='Card.TFrame', padding=12)
        ttk.Label(self.description_card, text='Scene understanding', style='Card.TLabel', font=self.bold_font).pack(anchor='w')
        wrap_label(ttk.Label(self.description_card, textvariable=self.description_title, style='Card.TLabel', foreground=MUTED)).pack(fill='x', pady=(4, 10))
        body = ttk.Frame(self.description_card, style='Card.TFrame'); body.pack(fill='both', expand=True)
        self.description_body = tk.Text(body, height=1, width=1, wrap='word', bg=PANEL, fg=FG,
                                       font=self.body_font, relief='flat', borderwidth=0, state='disabled', padx=2, pady=4, spacing1=2, spacing3=5)
        self.description_body.pack(side='left', fill='both', expand=True)
        scroll = ttk.Scrollbar(body, command=self.description_body.yview); scroll.pack(side='right', fill='y')
        self.description_body.configure(yscrollcommand=scroll.set)
        self.description_text.trace_add('write', lambda *_: self._description_updated())
        self._description_updated()
        self.flow_rows = (tools, model, actions)
        for widget in (self.source_combo,self.source_entry,self.camera_combo,self.scale_combo,self.device_combo,self.mode_combo):
            widget.configure(font=self.body_font)
        self._arrange_workspace(1440, 900)
        self._draw_preview()

    def _arrange_workspace(self, width, height):
        mode = 'wide' if width >= 1100 or height < 700 else 'stacked'
        if mode == self.workspace_mode:
            return
        self.workspace_mode = mode
        self.preview_card.grid_forget(); self.description_card.grid_forget()
        self.workspace.columnconfigure(0, weight=3 if mode == 'wide' else 1, uniform='workspace' if mode == 'wide' else '')
        self.workspace.columnconfigure(1, weight=2 if mode == 'wide' else 0, uniform='workspace' if mode == 'wide' else '')
        self.workspace.rowconfigure(0, weight=1 if mode == 'wide' else 3)
        self.workspace.rowconfigure(1, weight=0 if mode == 'wide' else 2)
        self.preview_card.grid(row=0, column=0, sticky='nsew', padx=(0, 12) if mode == 'wide' else 0)
        self.description_card.grid(row=0 if mode == 'wide' else 1, column=1 if mode == 'wide' else 0,
                                   sticky='nsew', pady=0 if mode == 'wide' else (10, 0))

    def _description_updated(self):
        self.description_body.configure(state='normal')
        self.description_body.delete('1.0','end')
        self.description_body.insert('1.0',self.description_text.get())
        self.description_body.tag_configure('section',font=self.bold_font,foreground=ACCENT)
        self.description_body.tag_configure('take_action',font=self.bold_font,foreground=AMBER,background='#3b3021')
        self.description_body.tag_configure('no_action',font=self.bold_font,foreground=ACCENT,background='#173b34')
        for index, line in enumerate(self.description_text.get().splitlines(), 1):
            for heading in ('Detections:', 'Environment:', 'Decision:', 'Reason:'):
                if line.startswith(heading):
                    self.description_body.tag_add('section', f'{index}.0', f'{index}.{len(heading)}')
            if line in ('Decision: Take Action', 'Decision: No Action'):
                self.description_body.tag_add('take_action' if line.endswith('Take Action') else 'no_action',
                                               f'{index}.10', f'{index}.end')
        self.description_body.configure(state='disabled')

    def _window_resized(self, event):
        if event.widget is not self.root or self.closing:
            return
        size = (event.width, event.height)
        if size == self.layout_size:
            return
        self.layout_size = size
        if self.layout_job:
            self.root.after_cancel(self.layout_job)
        self.layout_job = self.root.after(40, self._adapt_layout)

    def _adapt_layout(self):
        self.layout_job = None
        width, height = self.layout_size
        compact = height < 700 or width < 1000
        self.heading_label.configure(text='CNN + LLM Descriptor' if compact else 'CNN + LLM Based Descriptor')
        self.title_font.configure(size=16 if compact else 22)
        self._arrange_workspace(width, height)
        size = 11 if width >= 1100 and height >= 700 else 10
        if size != self.font_size:
            self.font_size = size
            self.body_font.configure(size=size)
            self.bold_font.configure(size=size)
            ttk.Style().configure('TButton',padding=(12, 8))
            self._draw_preview()
        for row in self.flow_rows:
            row.reflow()

    def _combo(self, parent, label, key, values, callback=None):
        ttk.Label(parent, text=label).pack(anchor='w', pady=(10,3))
        widget = ttk.Combobox(parent, textvariable=self.vars[key], values=values, state='readonly',font=self.body_font)
        widget.pack(fill='x')
        if callback:
            widget.bind('<<ComboboxSelected>>', lambda _e: callback())
        return widget

    def _source_changed(self, reset=True):
        kind = self.vars['source_kind'].get()
        camera = kind in {'Webcam','Pi camera (CSI)'}
        named = kind=='Webcam' and self.named_cameras
        self.source_label.configure(text='Camera' if named else 'Camera index (usually 0)' if camera else 'Stream URL' if kind=='Network stream' else 'Existing video to play' if kind=='Video' else 'Existing image to open')
        for widget in (self.source_entry,self.camera_combo,self.browse_button,self.refresh_button):
            widget.grid_remove()
        (self.camera_combo if named else self.source_entry).grid()
        if named:
            self.refresh_button.grid()
        elif kind in {'Video','Image'}:
            self.browse_button.grid()
        if reset:
            self.vars['source'].set('0' if camera else '')
        if named:
            self.refresh_cameras()
        self.start_button.configure(text='Open camera' if camera else 'Open stream' if kind=='Network stream' else 'Start')

    def refresh_cameras(self):
        if self.worker or not self.named_cameras:
            return
        devices, errors = discover_cameras()
        self.camera_map = {device['label']:device['path'] for device in devices}
        self.camera_combo.configure(values=list(self.camera_map))
        previous = linux_camera_path(self.vars['source'].get()).resolve()
        selected = next((label for label,path in self.camera_map.items() if Path(path).resolve()==previous),next(iter(self.camera_map),None))
        self.selected_camera.set(selected or 'No camera found — reconnect and Refresh')
        self.vars['source'].set(self.camera_map.get(selected,''))
        if errors:
            self.status.set(errors[0]); self.log('\n'.join(errors))
        elif not devices:
            self.status.set('No camera found. Reconnect your camera and click Refresh.')
        else:
            self.status.set('Camera ready · '+selected)
        self._controls()

    def _camera_selected(self):
        self.vars['source'].set(self.camera_map.get(self.selected_camera.get(),''))

    def _controls(self):
        running = self.worker is not None
        stopping = self.stopping_since is not None
        self.start_button.configure(state='disabled' if running or self.probe else 'normal')
        self.stop_button.configure(state='normal' if running and not stopping else 'disabled')
        for widget in (self.source_combo, self.scale_combo, self.device_combo,self.camera_combo):
            widget.configure(state='disabled' if running else 'readonly')
        self.source_entry.configure(state='disabled' if running else 'normal')
        self.browse_button.configure(state='disabled' if running else 'normal')
        self.refresh_button.configure(state='disabled' if running else 'normal')
        self.settings_button.configure(state='disabled' if running or self.probe else 'normal')
        self.inference_button.configure(state='disabled' if stopping else 'normal')
        self.mode_combo.configure(state='disabled' if self.recording or self.record_pending or stopping else 'readonly')
        can_record = running and self.ready and not stopping and not self.record_pending and self.vars['source_kind'].get()!='Image'
        self.record_button.configure(state='normal' if can_record else 'disabled', text='■ Stop recording' if self.recording else '● Record')
        for row in self.flow_rows:
            row.reflow()

    def _config(self):
        data = {key: value.get() for key,value in self.vars.items()}
        for key in ('image_size','max_detections','frame_stride','cpu_threads','camera_width','camera_height','camera_fps','describe_max_objects','describe_max_tokens','describe_image_size'):
            data[key] = int(data[key])
        for key in ('confidence','describe_interval','describe_timeout'):
            data[key] = float(data[key])
        device = self.device_map.get(self.selected_device.get())
        if device is None and data['inference_enabled']:
            raise ValueError('Select an available GPU or CPU, or turn inference off.')
        data['device'] = device or 'cpu'
        return RunConfig(**data)

    def refresh_devices(self):
        if self.worker or self.probe:
            return
        self.probe_queue = self.ctx.Queue()
        self.probe = self.ctx.Process(target=probe_devices, args=(self.probe_queue,), daemon=True)
        self.probe_started = time.monotonic()
        self.probe.start(); self._controls()

    def _devices(self, packet):
        self.device_map = {f"GPU {i} · {d['name']}": d['id'] for i,d in enumerate(packet['devices'])}
        self.device_map['CPU'] = 'cpu'
        self.device_combo.configure(values=list(self.device_map))
        preferred = self.vars['device'].get()
        self.selected_device.set(next((name for name,value in self.device_map.items() if value==preferred), 'CPU'))
        self.system_text = f"PyTorch {packet['torch']} · CUDA {packet['cuda']} · Ultralytics {packet['ultralytics']}\n"+'\n'.join(self.device_map)
        self.status.set('Ready · open a camera, then start recording when needed')
        if self.named_cameras and self.vars['source_kind'].get()=='Webcam' and not self.camera_map:
            self.status.set('No camera found. Reconnect your camera and click Refresh.')

    def inference_changed(self):
        enabled = self.vars['inference_enabled'].get()
        if self.worker:
            (self.inference_switch.set if enabled else self.inference_switch.clear)()
            self.status.set('Inference on · applies after the current frame' if enabled else 'Inference off · original preview')
        if not enabled:
            self.description_title.set('Visual description · inference off')
            self.description_text.set('Enable YOLO inference to generate descriptions.')
        if not enabled and not self.recording:
            self.vars['record_mode'].set('Original')
        self._controls()

    def start(self):
        if self.worker or self.probe or self.closing:
            return
        if self.named_cameras and self.vars['source_kind'].get()=='Webcam':
            self.refresh_cameras()
        try:
            cfg = self._config(); cfg.validated_source()
        except (ValueError,tk.TclError) as exc:
            messagebox.showerror('Check settings', str(exc)); return
        recording_path = None
        if cfg.save_output and cfg.source_kind != 'Image':
            recording_path = self._choose_recording_file(cfg.record_mode if cfg.inference_enabled else 'Original')
            if not recording_path:
                cfg = replace(cfg, save_output=False)
                self.vars['save_output'].set(False)
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.destroy()
        self._save_settings(cfg)
        self.messages, self.previews, self.commands = self.ctx.Queue(), self.ctx.Queue(2), self.ctx.Queue()
        self.stop_event, self.pause_event, self.inference_switch = self.ctx.Event(), self.ctx.Event(), self.ctx.Event()
        if cfg.inference_enabled:
            self.inference_switch.set()
        self.ready = self.recording = self.record_pending = False
        self.outcome = self.last_output = self.stopping_since = None
        self.last_error = None
        self.image = None; self._draw_preview()
        self.snapshot_button.configure(state='disabled')
        self.latest_records = []
        self.latest_description = None
        self.description_title.set('Visual description · waiting for a frame' if cfg.describe_enabled else 'Visual description · disabled')
        self.description_text.set(('CIS 579 probability and search are ready; waiting for YOLO detections.' if cfg.describe_backend == 'CIS 579' else 'The local model may take a moment to load.') if cfg.describe_enabled else 'Enable descriptions in Settings.')
        self.record_status.set('Not recording')
        self.record_label.configure(foreground=MUTED)
        self.worker = self.ctx.Process(target=self.worker_target, args=(cfg,self.messages,self.previews,self.stop_event,self.pause_event,self.commands,self.inference_switch,recording_path), daemon=True)
        self.worker.start(); self._controls()
        self.log('Opening '+cfg.source_kind)

    def toggle_record(self):
        if not self.worker or not self.ready or self.record_pending:
            return
        mode = self.vars['record_mode'].get()
        if not self.recording and mode!='Original' and not self.vars['inference_enabled'].get():
            messagebox.showinfo('Recording mode', 'Enable inference for annotated recording, or choose Original.'); return
        path = None
        if not self.recording:
            path = self._choose_recording_file(mode)
            if not path or not self.worker or not self.ready or self.stopping_since:
                return
        self.record_pending = True
        self.commands.put({'record':not self.recording,'mode':mode,'path':path})
        self.record_status.set('Finishing recording…' if self.recording else 'Starting recording…')
        self._controls()

    def stop(self):
        if self.worker and self.stopping_since is None:
            self.stop_event.set(); self.pause_event.clear()
            self.stopping_since = time.monotonic()
            self.status.set('Stopping and finalizing recording…'); self._controls()

    def _handle(self, packet):
        kind = packet['type']
        if kind=='description':
            if not self.vars['inference_enabled'].get() or self.stopping_since:
                return
            self.latest_description = packet
            position = f" · source {packet['source_seconds']:.1f}s" if packet.get('source_seconds') is not None else ''
            method = packet['model'] if packet.get('grounding_status', 'validated') == 'validated' else 'YOLO facts only'
            self.description_title.set(f"Frame {packet['frame']}{position} · {method} · {packet['elapsed_seconds']:.1f}s")
            self.description_text.set(packet['text'])
            self.log(f"Description for frame {packet['frame']}: {packet['text']}")
        elif kind in {'description_error','description_save_error'}:
            self.description_title.set('Visual description · unavailable' if kind=='description_error' else 'Visual description · save failed')
            self.description_text.set(packet['message'])
            self.log(packet['message'])
        elif kind=='reconnecting':
            self.description_title.set('Visual description · camera reconnecting')
            self.description_text.set('Waiting for a new frame.')
            self.ready = False
            self.status.set(packet['message']); self.log(packet['message']); self._controls()
        elif kind in {'status','ready','model'}:
            if not self.stopping_since:
                self.status.set(packet['message'])
            self.log(packet['message'])
            if kind=='ready':
                self.ready = True; self._controls()
            if packet.get('names'):
                self.system_text += '\nClasses: '+', '.join(f'{i}: {name}' for i,name in packet['names'].items())
        elif kind=='output':
            files = packet.get('video_files', {})
            self.last_output = Path(next(iter(files.values()))).parent if files else Path(packet['path'])
        elif kind=='recording':
            self.recording, self.record_pending = packet['active'], False
            if not self.recording and not self.vars['inference_enabled'].get():
                self.vars['record_mode'].set('Original')
            self.record_status.set('● REC · '+packet.get('mode','') if self.recording else 'Recording saved')
            self.record_label.configure(foreground='#ff7186' if self.recording else MUTED)
            self.log(packet['message']); self._controls()
        elif kind in {'error','record_error','inference_error'}:
            if kind=='error':
                self.outcome = 'failed'
                self.last_error = packet['message'].splitlines()[0]
            elif kind=='inference_error':
                self.vars['inference_enabled'].set(False)
                if not self.recording:
                    self.vars['record_mode'].set('Original')
            else:
                self.recording = self.record_pending = False
                self.record_status.set('Recording failed · see Details')
                self.record_label.configure(foreground='#ff7186')
            self.log(packet['message']+'\n'+packet.get('traceback',''))
            self.status.set(packet['message'].splitlines()[0][:150]+' · see Details')
            self._controls()
        elif kind=='done':
            self.outcome = packet['outcome']
            self.log(f"Session {self.outcome} · {packet['frames']} frames")

    def _preview(self, packet):
        self.last_packet = packet
        self.image = Image.fromarray(packet['rgb']); self._draw_preview()
        self.snapshot_button.configure(state='normal')
        self.latest_records = packet['records']
        inference = f"{packet['count']} predictions · {packet['inference_ms']:.0f} ms" if packet.get('inference',True) else 'Inference off'
        self.stats.set(f"{packet['fps']:.1f} FPS  ·  {packet['resolution']}  ·  {inference}")

    def _resize(self, _event=None):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(80,self._draw_preview)

    def _draw_preview(self):
        self.resize_job = None
        if not hasattr(self,'canvas'): return
        w,h = self.canvas.winfo_width(),self.canvas.winfo_height()
        self.canvas.delete('all')
        if self.image is None:
            self.canvas.create_text(w//2,h//2 if h < 100 else h//2-16,text='Your camera, live.',fill=FG,font=self.bold_font if h < 150 else self.title_font,width=max(100,w-30))
            if h < 100: return
            self.canvas.create_text(w//2,h//2+22,text='Open a camera · Turn inference on or off · Record',fill=MUTED,font=self.body_font,width=max(100,w-30))
        else:
            ratio = min(max(1,w-8)/self.image.width,max(1,h-8)/self.image.height)
            display = self.image.resize((max(1,round(self.image.width*ratio)),max(1,round(self.image.height*ratio))),Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(display)
            self.canvas.create_image(w//2,h//2,image=self.photo)

    def _pump(self):
        try:
            while True:
                self.model_status.set(self.model_messages.get_nowait())
        except queue.Empty:
            pass
        if self.probe:
            alive = self.probe.is_alive()
            try:
                packet = self.probe_queue.get_nowait()
                if packet['type']=='devices': self._devices(packet)
                else:
                    self.selected_device.set('CPU')
                    self.status.set('Inference setup needs attention · raw camera mode is available')
                    self.log(packet['message'])
            except queue.Empty: pass
            if alive and time.monotonic()-self.probe_started>45:
                self.probe.kill()
            if not alive:
                self.probe.join(0); self.probe.close(); self.probe = None
                self.probe_queue.close(); self._controls()
        if self.worker:
            alive = self.worker.is_alive()
            try:
                while True: self._handle(self.messages.get_nowait())
            except queue.Empty: pass
            packet = None
            try:
                while True: packet = self.previews.get_nowait()
            except queue.Empty: pass
            if packet is not None: self._preview(packet)
            if self.stopping_since and alive and time.monotonic()-self.stopping_since>8:
                self.worker.kill(); self.outcome = 'force-stopped'; self.stopping_since = None
                self.log('Forced stop: the active MP4 may be incomplete.')
            if not alive:
                self.worker.join(0); self.worker.close(); self.worker = None
                for channel in (self.messages,self.previews,self.commands): channel.close()
                self.ready = self.recording = self.record_pending = False
                self.stopping_since = None
                self.record_label.configure(foreground=MUTED)
                if self.record_status.get().startswith(('● REC','Starting','Finishing')):
                    self.record_status.set('Not recording')
                if not self.outcome:
                    self.outcome = 'failed'; self.log('Worker exited unexpectedly. Check memory and dependencies.')
                self.status.set('Stopped' if self.outcome=='stopped' else 'Complete' if self.outcome=='completed' else self.last_error or 'Session failed · see Details')
                self._controls()
        if self.closing and self.worker is None:
            self._destroy(); return
        self.timer = self.root.after(40,self._pump)

    def show_settings(self):
        if self.worker: return
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift(); return
        window = self.settings_window = tk.Toplevel(self.root)
        window.title('Settings'); resizable_window(window,880,700,minimum=(640,480)); window.configure(bg=BG)
        window.transient(self.root)
        ttk.Button(window,text='Done',command=window.destroy).pack(side='bottom',pady=(0,12))
        book = ttk.Notebook(window); book.pack(fill='both',expand=True,padx=14,pady=14)
        pages = [ScrollFrame(book, padding=14) for _ in range(4)]
        for name, page in zip(('Model', 'Camera', 'Saving', 'Descriptions'), pages): book.add(page, text=name)
        model,camera,output,description = (page.body for page in pages)
        def cell(parent,row,column,span=1):
            parent.columnconfigure(column,weight=1)
            frame=ttk.Frame(parent); frame.grid(row=row,column=column,columnspan=span,sticky='ew',padx=(0,8),pady=(0,4))
            return frame
        self._combo(cell(model,0,0),'Task','task',list(TASKS))
        self._combo(cell(model,0,1),'Inference size','image_size',[320,416,512,640,768,960,1280])
        self._field(cell(model,1,0),'Confidence','confidence')
        self._field(cell(model,1,1),'Maximum detections','max_detections')
        self._field(cell(model,2,0,2),'Class IDs (blank = all)','classes')
        self._field(cell(model,3,0,2),'Custom .pt weights (optional)','custom_model')
        options=FlowRow(cell(model,4,0,2)); options.pack(fill='x')
        options.add(ttk.Button(options,text='Browse model…',command=self.browse_model))
        options.add(ttk.Checkbutton(options,text='FP16 on GPU',variable=self.vars['half']))
        options.add(ttk.Checkbutton(options,text='Show labels',variable=self.vars['labels']))
        for i,(label,key) in enumerate([('Requested width','camera_width'),('Requested height','camera_height'),('Requested camera FPS','camera_fps'),('Inference frame stride','frame_stride'),('CPU threads','cpu_threads')]):
            self._field(cell(camera,i//2,i%2),label,key)
        ttk.Button(cell(camera,3,0,2),text='Raspberry Pi / SBC preset',command=self.apply_sbc_preset).pack(fill='x',pady=10)
        ttk.Label(cell(camera,4,0,2),text='Capture size and FPS depend on your camera.\nThe SBC preset uses Nano, 320 px, and CPU.',foreground=MUTED).pack(anchor='w')
        self._field(output,'Default save folder (optional)','output_dir')
        ttk.Button(output,text='Choose folder…',command=self.browse_output).pack(anchor='w',pady=6)
        ttk.Checkbutton(output,text='Record from start / save image result',variable=self.vars['save_output']).pack(anchor='w',pady=8)
        wrap_label(ttk.Label(output,text='A video filename is requested only when recording starts. Cancel keeps the preview available.\n\nOriginal contains no overlays. Both saves the annotated video at your chosen filename and a second file ending in _original.mp4.\n\nLive clips keep elapsed time by repeating frames when inference is slower than capture. Videos are silent.',foreground=MUTED)).pack(fill='x',pady=8)

        ttk.Checkbutton(description,text='Enable scene descriptions',variable=self.vars['describe_enabled']).pack(anchor='w')
        self._combo(description,'Description method','describe_backend',DESCRIBE_BACKENDS)
        self._field(description,'Minimum seconds between frames','describe_interval')
        wrap_label(ttk.Label(description,text='LLM uses the image to estimate the setting: home, ballroom, office, street, and more. CIS 579 uses only detected objects, conditional probability, rules, and uniform-cost search; no LLM or Ollama is needed.',foreground=MUTED)).pack(fill='x',pady=8)
        wrap_label(ttk.Label(description,text='Both methods keep object facts tied to YOLO. Settings are estimates. CIS 579 probabilities come from an illustrative table, not measured accuracy. Details shows the probabilities and review plan.',foreground=MUTED)).pack(fill='x',pady=8)
        ttk.Separator(description).pack(fill='x',pady=8)
        ttk.Label(description,text='LLM settings (used only in LLM mode)',font=self.bold_font).pack(anchor='w')
        for label,key in [('Quantized vision model','describe_model'),('Local Ollama server','describe_url'),('Request timeout (seconds)','describe_timeout')]:
            self._field(description,label,key)
        wrap_label(ttk.Label(description,text=f'Model folder: {MODEL_DIRECTORY}',foreground=MUTED)).pack(fill='x',pady=8)
        buttons = FlowRow(description); buttons.pack(fill='x')
        buttons.add(ttk.Button(buttons, text='Download model', command=self.download_description_model))
        buttons.add(ttk.Button(buttons, text='Cancel download', command=self.model_cancel.set))
        buttons.add(ttk.Button(buttons, text='Get Ollama', command=lambda: webbrowser.open('https://ollama.com/download')))
        wrap_label(ttk.Label(description, textvariable=self.model_status, foreground=ACCENT)).pack(fill='x', pady=6)

    def download_description_model(self):
        if self.model_job and self.model_job.is_alive():
            return
        try:
            cfg = replace(self._config(), describe_enabled=True, describe_backend="LLM")
            cfg.validate_descriptor()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Check description settings', str(exc)); return
        self.model_cancel.clear()
        self.model_status.set('Downloading vision model…')
        def progress(message):
            if self.model_messages.full():
                try: self.model_messages.get_nowait()
                except queue.Empty: pass
            self.model_messages.put_nowait(message)
        def work():
            try:
                download_model(cfg.describe_url, cfg.describe_model, progress, self.model_cancel)
                progress('Model ready · ' + cfg.describe_model)
            except Exception as exc:
                progress('Download needs attention · ' + str(exc))
        self.model_job = threading.Thread(target=work, name='model-download', daemon=True)
        self.model_job.start()

    def _field(self,parent,label,key):
        ttk.Label(parent,text=label).pack(anchor='w',pady=(7,2))
        ttk.Entry(parent,textvariable=self.vars[key],font=self.body_font).pack(fill='x')

    def apply_sbc_preset(self):
        for key,value in {'scale':'N · Nano','device':'cpu','image_size':320,'half':False,'cpu_threads':2,'frame_stride':2,'camera_width':640,'camera_height':480,'camera_fps':30}.items(): self.vars[key].set(value)
        self.selected_device.set('CPU')

    def show_details(self):
        if self.details_window and self.details_window.winfo_exists(): self.details_window.destroy()
        window = self.details_window = tk.Toplevel(self.root); window.title('Session details'); resizable_window(window,900,600,minimum=(600,400)); window.transient(self.root)
        text = tk.Text(window,bg=PANEL,fg=FG,wrap='word',font=('DejaVu Sans Mono',9)); text.pack(fill='both',expand=True)
        text.insert('end', self.system_text+'\n\nLatest description:\n'+json.dumps(self.latest_description,indent=2)+'\n\nLatest predictions:\n'+json.dumps(self.latest_records,indent=2)+'\n\nActivity:\n'+'\n'.join(self.log_lines))
        text.configure(state='disabled'); text.see('end')

    def log(self,text): self.log_lines.append(f'{datetime.now():%H:%M:%S}  {text}')

    def browse_source(self):
        value = filedialog.askopenfilename(title='Open existing video' if self.vars['source_kind'].get()=='Video' else 'Open existing image')
        if value: self.vars['source'].set(value)

    def _choose_recording_file(self, mode):
        directory = Path(self.vars['output_dir'].get()).expanduser()
        while True:
            value = filedialog.asksaveasfilename(parent=self.root,
                title='Save annotated video (also saves original)' if mode=='Both' else 'Save recording',
                initialdir=str(directory if directory.is_dir() else ROOT),
                initialfile=f'recording_{datetime.now():%Y%m%d_%H%M%S}.mp4',
                defaultextension='.mp4', filetypes=[('MP4 video','*.mp4')], confirmoverwrite=False)
            if not value:
                return None
            try:
                details, paths = recording_destinations(value, mode)
                if any(path.exists() for path in (details, *paths.values())):
                    raise ValueError('Recording files already use that name. Choose a new filename.')
            except ValueError as exc:
                messagebox.showerror('Choose a recording filename',str(exc),parent=self.root)
                continue
            self.vars['output_dir'].set(str(Path(value).parent))
            return value

    def browse_model(self):
        value = filedialog.askopenfilename(title='Choose trusted model',filetypes=[('PyTorch weights','*.pt')])
        if value: self.vars['custom_model'].set(value)

    def browse_output(self):
        value = filedialog.askdirectory(title='Output folder')
        if value: self.vars['output_dir'].set(value)

    def snapshot(self):
        if self.image is None: return
        path = filedialog.asksaveasfilename(defaultextension='.png',initialfile=f'snapshot_{datetime.now():%Y%m%d_%H%M%S}.png',filetypes=[('PNG','*.png')])
        if path:
            try: self.image.save(path)
            except OSError as exc: messagebox.showerror('Could not save snapshot',str(exc))

    def open_outputs(self):
        path = self.last_output or Path(self.vars['output_dir'].get()).expanduser()
        try:
            path.mkdir(parents=True,exist_ok=True)
            if sys.platform=='win32': os.startfile(str(path.resolve()))
            else: subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',str(path.resolve())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except OSError as exc: messagebox.showerror('Could not open outputs',str(exc))

    def _save_settings(self,cfg):
        data = asdict(cfg)
        if cfg.source_kind=='Network stream': data['source']=''
        try:
            temporary = self.settings_path.with_suffix('.tmp'); temporary.write_text(json.dumps(data,indent=2)); temporary.replace(self.settings_path)
        except OSError as exc: self.log(str(exc))

    def close(self):
        self.closing = True
        self.model_cancel.set()
        try: self._save_settings(self._config())
        except (ValueError,tk.TclError): pass
        if self.worker: self.stop()

    def _destroy(self):
        if self.probe:
            self.probe.kill(); self.probe.join(.2)
        for job in (self.timer,self.resize_job,self.layout_job):
            if job: self.root.after_cancel(job)
        self.root.destroy()


def main():
    root = tk.Tk(); App(root); root.mainloop()
