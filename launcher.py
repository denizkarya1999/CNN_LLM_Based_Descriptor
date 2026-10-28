#!/usr/bin/env python3
"""Standard-library GUI for setup and launch; no command-line arguments needed."""

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import uuid
from cnn_llm_based_descriptor.window import resizable_window, FlowRow, ScrollFrame, wrap_label
from cnn_llm_based_descriptor.branding import apply_icon
from cnn_llm_based_descriptor.theme import apply_theme, BG, PANEL, FG, MUTED, ACCENT

ROOT = Path(__file__).resolve().parent
REUSE = "Use installed Python packages"
CPU = "CPU · Raspberry Pi / SBC / PC"
CUDA = "NVIDIA GPU · CUDA"
CUDA_INDEXES = {"CUDA 13.0": "https://download.pytorch.org/whl/cu130",
                "CUDA 12.8": "https://download.pytorch.org/whl/cu128"}
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

PROBE = r'''
import json, re, sys
try:
    import torch, torchvision, ultralytics, cv2, PIL, numpy, tkinter
    version = tuple(int(n) for n in re.findall(r"\d+", ultralytics.__version__)[:3])
    if version < (8, 4, 56):
        raise RuntimeError("Ultralytics 8.4.56 or newer is required for this project.")
    devices = []
    gpu_warning = ""
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.empty(1, device=f"cuda:{i}")
                devices.append({"id": f"cuda:{i}", "name": torch.cuda.get_device_name(i)})
    except Exception as exc:
        gpu_warning = str(exc)
    report = {"ok": True, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
              "cuda_build": torch.version.cuda, "devices": devices, "gpu_warning": gpu_warning}
except Exception as exc:
    report = {"ok": False, "error": str(exc)}
print("CNN_LLM_BASED_DESCRIPTOR_PROBE=" + json.dumps(report), flush=True)
sys.exit(0 if report["ok"] else 1)
'''


class Cancelled(Exception):
    pass


def env_python(directory):
    return Path(directory) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def active_environment(project):
    """Accept only project-owned environment directories from saved preferences."""
    project = Path(project).resolve()
    try:
        entry = json.loads((project / "launcher_settings.json").read_text())["environment"]
        directory = (project / entry).resolve()
        if directory == project / ".venv" or directory.parent == project / ".environments":
            return directory
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return project / ".venv"


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_preferences(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class CommandRunner:
    """Stream subprocess output without blocking the GUI; cancel the whole job."""

    def __init__(self, project, cancelled, emit):
        self.project, self.cancelled, self.emit = Path(project), cancelled, emit

    def run(self, args, timeout=None, check=True, visible=True):
        if self.cancelled.is_set():
            raise Cancelled()
        environment = os.environ.copy()
        config_dir = self.project / ".ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        environment["YOLO_CONFIG_DIR"] = str(config_dir)
        environment["MPLCONFIGDIR"] = str(config_dir / "matplotlib")
        environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        environment["PIP_NO_INPUT"] = "1"
        process = subprocess.Popen([str(a) for a in args], cwd=self.project, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, errors="replace", start_new_session=(os.name != "nt"))
        lines = []

        def read_output():
            for line in process.stdout:
                lines.append(line)
                if len(lines) > 2000:
                    del lines[:500]
                if visible:
                    self.emit("log", line.rstrip())

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        start = time.monotonic()
        try:
            while process.poll() is None:
                if self.cancelled.is_set():
                    raise Cancelled()
                if timeout and time.monotonic() - start > timeout:
                    raise RuntimeError("The installation check timed out. Check your Python installation and drivers.")
                time.sleep(.1)
        finally:
            if process.poll() is None:
                try:
                    if os.name == "nt":
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            reader.join(timeout=5)
            process.stdout.close()
        if self.cancelled.is_set():
            raise Cancelled()
        output = "".join(lines)
        if check and process.returncode:
            raise RuntimeError("The operation failed. See the log below.\n" + output[-1800:])
        return process.returncode, output


def probe_python(python, runner):
    if not Path(python).is_file():
        return {"ok": False, "error": "No configured Python environment yet. Choose a setup option below."}
    _, output = runner.run([python, "-c", PROBE], timeout=60, check=False, visible=False)
    for line in reversed(output.splitlines()):
        if line.startswith("CNN_LLM_BASED_DESCRIPTOR_PROBE="):
            return json.loads(line.split("=", 1)[1])
    return {"ok": False, "error": "Python could not complete its dependency check. " + output[-1200:]}


def build_environment(project, mode, cuda_index, runner):
    """Build in a new permanent directory; keep the active environment on failure."""
    project = Path(project)
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("Install a 64-bit OS and Python before setting up YOLO.")
    if mode not in {REUSE, CPU, CUDA}:
        raise ValueError("Choose a setup option.")
    if cuda_index not in CUDA_INDEXES:
        raise ValueError("Choose one of the listed CUDA builds.")
    directory = project / ".environments" / (datetime.now().strftime("env-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    directory.parent.mkdir(exist_ok=True)
    try:
        runner.emit("log", "Creating a separate Python environment…")
        command = [sys.executable, "-m", "venv"]
        if mode == REUSE:
            command.extend(["--system-site-packages", "--without-pip"])
        elif platform.machine().lower() in {"arm64", "aarch64"}:
            command.append("--system-site-packages")  # Access OS-provided Picamera2/libcamera.
        command.append(str(directory))
        runner.run(command)
        python = env_python(directory)
        if mode != REUSE:
            runner.emit("log", "Installing Python package tools…")
            runner.run([python, "-m", "pip", "install", "--upgrade", "pip", "--progress-bar", "off"])
            runner.emit("log", "Installing the selected PyTorch build. Downloads may take several minutes…")
            index = CUDA_INDEXES[cuda_index] if mode == CUDA else CPU_INDEX
            # On ARM, system packages are visible. Ignore them when installing the
            # requested torch build so a global CUDA build cannot satisfy CPU setup.
            runner.run([python, "-m", "pip", "install", "--ignore-installed", "torch", "torchvision",
                        "--index-url", index, "--progress-bar", "off"])
            runner.emit("log", "Installing the YOLO application packages…")
            runner.run([python, "-m", "pip", "install", "-r", project / "requirements.txt", "--progress-bar", "off"])
        runner.emit("log", "Verifying the new installation…")
        report = probe_python(python, runner)
        if not report["ok"]:
            raise RuntimeError(report["error"] + "\nChoose a fresh CPU/GPU installation, or install Linux prerequisites if needed.")
        if mode == CUDA and not report["devices"]:
            raise RuntimeError("PyTorch installed, but CUDA is unavailable. Check the NVIDIA driver or choose a CPU installation.")
        if mode == CPU and report["cuda_build"] is not None:
            raise RuntimeError("CPU setup unexpectedly found a CUDA PyTorch build; the existing environment remains active.")
        if runner.cancelled.is_set():
            raise Cancelled()
        write_json(project / "launcher_settings.json", {"environment": str(directory.relative_to(project)), "mode": mode})
        return directory, report
    except BaseException:
        if directory.exists():
            shutil.rmtree(directory)
        raise


class SetupWindow:
    def __init__(self, root, project=ROOT, on_launch=None):
        self.root, self.project = root, Path(project).resolve()
        self.on_launch = on_launch
        self.events = queue.Queue()
        self.cancelled = threading.Event()
        self.busy = self.closing = False
        self.job_cancellable = True
        self.ready = False
        self.environment = active_environment(self.project)
        self.device_map = {}
        self.status = tk.StringVar(value="Checking the installed environment…")
        self.device = tk.StringVar(value="Checking devices…")
        self.mode = tk.StringVar(value=REUSE if env_python(self.environment).is_file() else CPU)
        self.cuda = tk.StringVar(value="CUDA 13.0")
        self.csi = tk.BooleanVar(value=False)
        root.title("CNN + LLM Based Descriptor · Launch & Setup")
        resizable_window(root, 800, 740, minimum=(640, 480))
        apply_theme(root)
        self.brand_icon = apply_icon(root)
        header = ttk.Frame(root, padding=(20, 18, 20, 12))
        header.pack(fill='x')
        wrap_label(ttk.Label(header, text='CNN + LLM Based Descriptor', image=self.brand_icon, compound='left', font=('DejaVu Sans', 17, 'bold'))).pack(fill='x')
        ttk.Label(header, text='Launch your workspace', foreground=MUTED).pack(anchor='w', pady=(4, 0))

        # Reserve the action area before packing scrollable content. Checks and
        # long logs can never consume the space needed to launch or cancel.
        footer = ttk.Frame(root, padding=(20, 10, 20, 16))
        footer.pack(side='bottom', fill='x')
        ttk.Separator(footer).pack(fill='x', pady=(0, 10))
        primary = FlowRow(footer); primary.pack(fill='x')
        self.open_button = primary.add(ttk.Button(primary, text='Open app', style='Open.TButton', state='disabled', command=self.launch), expand=True)
        self.check_button = primary.add(ttk.Button(primary, text='Check again', command=self.check))
        self.cancel_button = primary.add(ttk.Button(primary, text='Cancel operation', command=self.cancelled.set, state='disabled'))
        secondary = FlowRow(footer); secondary.pack(fill='x', pady=(8, 0))
        self.install_button = secondary.add(ttk.Button(secondary, text='Set up / repair', command=self.install), expand=True)
        self.system_button = secondary.add(ttk.Button(secondary, text='Linux prerequisites…', command=self.system_packages), expand=True)
        self.progress = ttk.Progressbar(footer, mode='indeterminate')
        self.progress.pack(fill='x', pady=(10, 0))

        self.form = ScrollFrame(root, padding=(20, 0, 16, 8))
        self.form.pack(fill='both', expand=True)
        content = self.form.body
        status_card = ttk.Frame(content, style='Card.TFrame', padding=14)
        status_card.pack(fill='x', pady=(0, 14))
        ttk.Label(status_card, text='INSTALLATION STATUS', style='Card.TLabel', foreground=ACCENT, font=('DejaVu Sans', 9, 'bold')).pack(anchor='w')
        wrap_label(ttk.Label(status_card, textvariable=self.status, style='Card.TLabel')).pack(fill='x', pady=(5, 0))
        ttk.Label(content, text='Processing device', font=('DejaVu Sans', 11, 'bold')).pack(anchor='w')
        self.devices = ttk.Combobox(content, textvariable=self.device, state='readonly')
        self.devices.pack(fill='x', pady=(6, 16))
        ttk.Separator(content).pack(fill='x', pady=(0, 14))
        self.setup_frame = setup = ttk.Frame(content)
        setup.pack(fill='x')
        self.setup_visible = True
        ttk.Label(setup, text='Installation & repair', font=('DejaVu Sans', 13, 'bold')).pack(anchor='w')
        wrap_label(ttk.Label(setup, text='Choose a package profile, then use Set up / repair below.', foreground=MUTED)).pack(fill='x', pady=(4, 8))
        self.profiles = ttk.Combobox(setup, textvariable=self.mode, values=[REUSE, CPU, CUDA], state='readonly')
        self.profiles.pack(fill='x')
        self.profiles.bind('<<ComboboxSelected>>', self._mode_changed)
        cuda_row = ttk.Frame(setup); cuda_row.pack(fill='x', pady=(10, 0))
        ttk.Label(cuda_row, text='CUDA package build').pack(side='left', padx=(0, 12))
        self.cuda_box = ttk.Combobox(cuda_row, textvariable=self.cuda, values=list(CUDA_INDEXES), state='disabled', width=18)
        self.cuda_box.pack(side='left')
        self.csi_check = ttk.Checkbutton(setup, text='Include Raspberry Pi CSI camera support', variable=self.csi)
        self.csi_check.pack(anchor='w', pady=(6, 8))
        ttk.Label(content, text='Activity', font=('DejaVu Sans', 11, 'bold')).pack(anchor='w', pady=(8, 6))
        log_frame = ttk.Frame(content); log_frame.pack(fill='x')
        self.log_widget = tk.Text(log_frame, height=5, bg=PANEL, fg=FG, relief='flat', wrap='word', font=('DejaVu Sans Mono', 9), state='disabled', padx=10, pady=8)
        self.log_widget.pack(side='left', fill='both', expand=True)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_widget.yview)
        log_scroll.pack(side='right', fill='y')
        self.log_widget.configure(yscrollcommand=log_scroll.set)
        wrap_label(ttk.Label(content, text='Fresh installs need internet. Linux prerequisites may request administrator authentication.', foreground=MUTED)).pack(fill='x', pady=(8, 0))
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = root.after(50, self._pump)
        self.check()

    def show_setup(self, visible):
        # Setup and its buttons remain available in every state. In particular,
        # a successful dependency check must not shrink a user-resized window.
        if visible:
            self.form.canvas.yview_moveto(0)

    def emit(self, kind, value):
        self.events.put((kind, value))

    def _mode_changed(self, _event=None):
        self.cuda_box.configure(state="readonly" if self.mode.get() == CUDA and not self.busy else "disabled")

    def _controls(self):
        for widget in (self.install_button, self.check_button, self.system_button, self.csi_check):
            widget.configure(state="disabled" if self.busy else "normal")
        self.profiles.configure(state="disabled" if self.busy else "readonly")
        self.devices.configure(state="disabled" if self.busy or not self.ready else "readonly")
        self.open_button.configure(state="normal" if self.ready and not self.busy else "disabled")
        self.cancel_button.configure(state="normal" if self.busy and self.job_cancellable else "disabled")
        self._mode_changed()

    def _start_job(self, job, status, cancellable=True):
        if self.busy:
            return
        self.busy = True
        self.job_cancellable = cancellable
        self.cancelled.clear()
        self.status.set(status)
        self._controls()
        self.progress.start(15)

        def work():
            try:
                job(CommandRunner(self.project, self.cancelled, self.emit))
            except Cancelled:
                self.emit("message", "Operation cancelled. The previously configured environment is unchanged.")
            except Exception as exc:
                self.emit("error", str(exc))
            finally:
                self.emit("finished", None)

        threading.Thread(target=work, daemon=True).start()

    def check(self):
        def job(runner):
            directory = active_environment(self.project)
            report = probe_python(env_python(directory), runner)
            self.emit("report", (directory, report, None))
        self._start_job(job, "Checking Python packages and available devices…")

    def install(self):
        mode, cuda = self.mode.get(), self.cuda.get()
        def job(runner):
            directory, report = build_environment(self.project, mode, cuda, runner)
            preferred = "cpu" if mode == CPU else (report["devices"][0]["id"] if mode == CUDA else None)
            self.emit("report", (directory, report, preferred))
        self._start_job(job, "Preparing your environment…")

    def system_packages(self):
        apt, pkexec = shutil.which("apt-get"), shutil.which("pkexec")
        if not apt or not pkexec:
            messagebox.showinfo("System prerequisites", "Use your operating system's software manager to install Python Tk, venv, pip, OpenGL/GLib libraries, and optionally Picamera2. This automatic installer requires apt-get and PolicyKit.")
            return
        packages = ["python3-tk", "python3-venv", "python3-pip", "libgl1", "libglib2.0-0"]
        if self.csi.get():
            packages.append("python3-picamera2")
        if not messagebox.askokcancel("Install system prerequisites", "Install these operating system packages?\n\n" + ", ".join(packages) + "\n\nYour system will request administrator authentication."):
            return
        def job(runner):
            runner.run([pkexec, apt, "install", "-y", *packages])
            self.emit("message", "Linux prerequisites installed. Choose a setup option and click Set up / repair.")
        # Interrupting dpkg halfway through can leave the OS package database
        # inconsistent. Closing waits for this explicit system operation to finish.
        self._start_job(job, "Installing Linux prerequisites · waiting for the system installer…", cancellable=False)

    def _report(self, directory, report, preferred):
        self.environment = directory
        self.ready = report["ok"]
        if not self.ready:
            self.status.set("Setup is needed · choose an option and click Set up / repair")
            self.show_setup(True)
            self.device.set("No ready environment")
            self.log(report["error"])
            return
        self.device_map = {f"{d['id']} · {d['name']}": d["id"] for d in report["devices"]}
        self.device_map["CPU · Raspberry Pi / SBC / PC"] = "cpu"
        self.devices.configure(values=list(self.device_map))
        if preferred is None:
            preferred = read_preferences(self.project / "settings.json").get("device")
        label = next((key for key, value in self.device_map.items() if value == preferred), next(iter(self.device_map)))
        self.device.set(label)
        self.status.set("Ready · choose a device and open the app")
        self.show_setup(False)
        self.log(f"PyTorch {report['torch']} · Ultralytics {report['ultralytics']}")
        if report.get("gpu_warning"):
            self.log("GPU check: " + report["gpu_warning"])
        self.log("Installation verified. Select GPU or CPU above, then Open CNN + LLM Based Descriptor.")

    def launch(self):
        if self.busy or not self.ready:
            return
        selected = self.device_map.get(self.device.get())
        if selected is None:
            return
        try:
            settings = read_preferences(self.project / "settings.json")
            if not settings and selected == "cpu" and platform.machine().lower() in {"arm64", "aarch64"}:
                settings.update(image_size=320, half=False, max_detections=50, frame_stride=2, cpu_threads=2)
            settings["device"] = selected
            write_json(self.project / "settings.json", settings)
            args = [str(env_python(self.environment)), str(self.project / "main.py")]
            if self.on_launch is not None:
                self.on_launch(args)
            else:
                with (self.project / "launcher.log").open("a") as output:
                    subprocess.Popen(args, cwd=self.project, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=(os.name != "nt"))
            self._destroy()
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("Could not launch", str(exc))

    def log(self, message):
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", str(message) + "\n")
        if int(self.log_widget.index("end-1c").split(".")[0]) > 1000:
            self.log_widget.delete("1.0", "250.0")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _pump(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.log(value)
                elif kind == "report":
                    self._report(*value)
                elif kind in {"error", "message"}:
                    self.status.set("Setup failed · see the log" if kind == "error" else value)
                    self.log(value)
                elif kind == "finished":
                    self.busy = False
                    self.progress.stop()
                    self._controls()
        except queue.Empty:
            pass
        if self.closing and not self.busy:
            self._destroy()
            return
        self.timer = self.root.after(50, self._pump)

    def _destroy(self):
        if self.timer:
            self.root.after_cancel(self.timer)
            self.timer = None
        self.root.destroy()

    def close(self):
        self.closing = True
        if self.busy:
            if self.job_cancellable:
                self.cancelled.set()
                self.status.set("Cancelling the current operation…")
            else:
                self.status.set("Waiting for the system installer to finish before closing…")
        else:
            self._destroy()


def main():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"CNN + LLM Based Descriptor needs a graphical desktop: {exc}", file=sys.stderr)
        return 1
    SetupWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
