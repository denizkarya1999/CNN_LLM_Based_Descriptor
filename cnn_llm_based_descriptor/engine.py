"""Inference runs in a separate process so GUI controls stay responsive."""

from pathlib import Path
import math
import os
import queue
import tempfile
import time
import traceback
import threading
from collections import deque
from dataclasses import replace

from .config import RunConfig, parse_classes, RECORD_MODES
from .recording import RunRecorder
from .cameras import open_webcam
from .descriptor import FrameDescriber


def probe_devices(messages) -> None:
    """Keep torch/CUDA initialization off the GUI thread."""
    try:
        import torch
        import ultralytics
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devices.append({"id": f"cuda:{i}", "name": p.name,
                                "memory_gb": round(p.total_memory / 1024**3, 1)})
        messages.put({"type": "devices", "devices": devices, "torch": torch.__version__,
                      "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__})
    except Exception as exc:
        messages.put({"type": "probe_error", "message": str(exc)})


def choose_device(requested: str, torch) -> str:
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check NVIDIA drivers and the CUDA-enabled PyTorch "
                           "installation, then Refresh GPU. You can explicitly select CPU to continue.")
    index = int(requested.split(":")[1])
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {index} is unavailable. Refresh GPU and select an available device.")
    # An allocation detects driver/runtime incompatibility before downloading a model.
    torch.empty(1, device=requested)
    return requested


class PiCameraCapture:
    """Optional Raspberry Pi CSI source, using the system Picamera2 package."""

    def __init__(self, index: int, width=640, height=480, fps=30):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError("Pi CSI cameras need Picamera2. Install python3-picamera2 with apt "
                               "and use a virtual environment with --system-site-packages. "
                               "USB cameras use Webcam instead.") from exc
        self.camera = Picamera2(camera_num=index)
        try:
            self.camera.configure(self.camera.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}, controls={"FrameRate": fps}, buffer_count=4))
            self.camera.start()
        except Exception:
            self.camera.close()
            raise

    def isOpened(self):
        return True

    def read(self):
        # Picamera2 RGB888 arrays are BGR byte order, matching OpenCV.
        return True, self.camera.capture_array("main")

    def grab(self):
        self.camera.capture_array("main")
        return True

    def get(self, _prop):
        return 0

    def set(self, _prop, _value):
        return False

    def release(self):
        try:
            self.camera.stop()
        finally:
            self.camera.close()


def publish_latest(channel, value) -> None:
    """Bound preview memory; saving/inference never wait for a slow GUI."""
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


def load_model(cfg, yolo_class):
    """Publish a cached checkpoint only after its download and load succeed."""
    if cfg.custom_model or cfg.weights.is_file():
        return yolo_class(str(cfg.weights))
    cfg.weights.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=cfg.weights.parent) as directory:
        staged = Path(directory) / cfg.weights.name
        model = yolo_class(str(staged))
        staged.replace(cfg.weights)
        model.ckpt_path = str(cfg.weights)
        return model


class LatestCamera:
    """Drain the camera continuously; inference always receives a recent frame."""

    def __init__(self, capture, stop):
        self.capture, self.stop = capture, stop
        self.condition = threading.Condition()
        self.closed = threading.Event()
        self.latest = None
        self.sequence = 0
        self.error = None
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.closed.is_set() and not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError("Camera or stream disconnected. Reconnect it, click Refresh for a webcam, then open it again.")
                with self.condition:
                    self.sequence += 1
                    self.latest = (self.sequence, frame, time.perf_counter())
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def next(self, after, timeout=6):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.stop.is_set():
                if self.latest is not None and self.latest[0] > after:
                    return self.latest
                if self.error:
                    raise RuntimeError(self.error)
                if time.monotonic() > deadline:
                    raise RuntimeError("No camera frame received. Close other camera apps, reconnect the camera, then click Refresh and Open camera.")
                self.condition.wait(.1)
        return None

    def release(self):
        self.closed.set()
        self.thread.join(.5)
        # Never release OpenCV while another thread is inside read(). A blocked
        # device is released by process exit; the GUI also has an 8-second stop limit.
        if not self.thread.is_alive():
            self.capture.release()


def open_capture(cfg, source):
    import cv2
    if cfg.source_kind == "Pi camera (CSI)":
        capture = PiCameraCapture(source, cfg.camera_width, cfg.camera_height, cfg.camera_fps)
    elif cfg.source_kind == 'Webcam':
        return open_webcam(cfg, source)
    elif cfg.source_kind == "Network stream":
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Could not open input. Check camera index/permissions, file, or stream connection.")
    return capture


def infer(cfg: RunConfig, messages, previews, stop, paused, commands=None, inference_switch=None, recording_path=None):
    """Preview and recording work without importing torch or loading a model."""
    import cv2
    capture = camera = recorder = model = torch = None
    options = {}
    describer = None
    was_enabled = False
    outcome = "completed"
    frame_index = processed = 0
    last_preview = 0.0
    completed_times = deque(maxlen=30)
    if inference_switch is None:
        inference_switch = threading.Event()
        if cfg.inference_enabled:
            inference_switch.set()

    def finish_recording(reason="completed"):
        nonlocal recorder
        current, recorder = recorder, None
        if current:
            try:
                current.close(reason, time.perf_counter())
                messages.put({"type": "recording", "active": False, "message": f"Recording {reason} · {current.directory}", "path": str(current.directory)})
            except Exception as exc:
                messages.put({"type": "record_error", "message": f"Could not finalize recording: {exc}"})

    def begin_recording(mode, path=None):
        nonlocal recorder
        if recorder:
            return
        if mode not in RECORD_MODES:
            messages.put({"type": "record_error", "message": "Choose a recording mode."})
            return
        if mode != "Original" and not inference_switch.is_set():
            messages.put({"type": "record_error", "message": "Enable inference to record overlays, or choose Original."})
            return
        try:
            recorder = RunRecorder(replace(cfg, inference_enabled=inference_switch.is_set()), fps if cfg.live else fps / cfg.frame_stride, mode, time.perf_counter(), path)
            messages.put({"type": "output", "path": str(recorder.directory), "video_files": {name: track.path for name, track in recorder.tracks.items()}})
            messages.put({"type": "recording", "active": True, "mode": mode, "message": f"Recording · {mode}"})
        except Exception as exc:
            messages.put({"type": "record_error", "message": f"Could not start recording: {exc}"})

    def reconnect_webcam():
        nonlocal camera, capture
        if describer:
            describer.invalidate()
        messages.put({'type': 'reconnecting', 'message': 'Camera disconnected · trying to reconnect for up to eight seconds…'})
        finish_recording('interrupted by camera disconnect')
        camera.release()
        camera = capture = None
        deadline = time.monotonic() + 8
        while not stop.wait(.5) and time.monotonic() < deadline:
            try:
                # Use the same by-id/by-path link; never switch to another camera.
                capture = open_capture(cfg, source)
                camera = LatestCamera(capture, stop)
                first = camera.next(0, timeout=max(.1, deadline-time.monotonic()))
                if first is None:
                    return None
                completed_times.clear()
                messages.put({'type': 'ready', 'message': 'Camera reconnected · click Record to start a new clip', 'names': {}})
                return first
            except (RuntimeError, OSError):
                if camera:
                    camera.release()
                elif capture:
                    capture.release()
                camera = capture = None
        if stop.is_set():
            return None
        raise RuntimeError('Camera could not reconnect. Check its connection or privacy switch, click Refresh, then Open camera.')

    try:
        source = cfg.validated_source()
        messages.put({"type": "status", "message": "Opening camera…" if cfg.live else "Opening input…"})
        fps, total, still = float(cfg.camera_fps), 1, None
        if cfg.source_kind == "Image":
            still = cv2.imread(str(source))
            if still is None:
                raise RuntimeError("Could not decode the image. Try a JPEG or PNG.")
        else:
            capture = open_capture(cfg, source)
            reported = capture.get(cv2.CAP_PROP_FPS)
            if math.isfinite(reported) and 0 < reported <= 240:
                fps = reported
            nframes = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            total = int(nframes) if not cfg.live and math.isfinite(nframes) and nframes > 0 else 0
            if cfg.live:
                camera = LatestCamera(capture, stop)
        messages.put({"type": "ready", "message": "Camera open" if cfg.live else "Input ready", "names": {}})
        if cfg.describe_enabled:
            describer = FrameDescriber(cfg, messages, stop, inference_switch)
        automatic_recording_pending = cfg.save_output
        while not stop.is_set():
            if commands is not None:
                try:
                    while True:
                        command = commands.get_nowait()
                        if command.get("record") is True:
                            begin_recording(command.get("mode", "Original"), command.get("path"))
                        elif command.get("record") is False:
                            finish_recording()
                except queue.Empty:
                    pass
            if recorder and recorder.mode != "Original" and not inference_switch.is_set():
                finish_recording("stopped because inference was turned off")
            if describer and was_enabled and not inference_switch.is_set():
                describer.invalidate()
                was_enabled = False
            if paused.is_set():
                if recorder:
                    finish_recording("paused")
                stop.wait(.05)
                continue
            if automatic_recording_pending:
                begin_recording(cfg.record_mode if cfg.inference_enabled else "Original", recording_path)
                automatic_recording_pending = False
            started = time.perf_counter()
            if still is not None:
                frame, still = still, None
                frame_index = 1
                captured_at = started
            elif camera is not None:
                try:
                    value = camera.next(frame_index)
                except RuntimeError:
                    if cfg.source_kind != 'Webcam':
                        raise
                    value = reconnect_webcam()
                if value is None:
                    break
                frame_index, frame, captured_at = value
            else:
                ok, frame = capture.read()
                if not ok:
                    if not processed:
                        raise RuntimeError("No video frames could be decoded.")
                    break
                frame_index += 1
                captured_at = started
            enabled = inference_switch.is_set()
            # In raw preview/recording mode, retain every acquired frame.
            if (enabled or not cfg.live) and (frame_index - 1) % cfg.frame_stride:
                continue
            records, inference_ms, allocated = [], 0.0, 0.0
            annotated = frame
            if enabled:
                try:
                    if model is None:
                        cfg.validate_inference()
                        messages.put({"type": "status", "message": f"Loading {cfg.weights.name}…"})
                        import torch as torch_module
                        from ultralytics import YOLO
                        from ultralytics.cfg import DEFAULT_CFG_DICT
                        torch = torch_module
                        torch.set_num_threads(min(cfg.cpu_threads, os.cpu_count() or 1))
                        device = choose_device(cfg.device, torch)
                        model = load_model(cfg, YOLO)
                        classes = parse_classes(cfg.classes) if model.task != "classify" else None
                        if classes and any(c not in model.names for c in classes):
                            raise ValueError(f"Class IDs must be between 0 and {len(model.names)-1}.")
                        options = {"device": device, "imgsz": cfg.image_size, "conf": cfg.confidence,
                            "max_det": cfg.max_detections, "classes": classes, "verbose": False, "save": False}
                        if "quantize" in DEFAULT_CFG_DICT:
                            options["quantize"] = 16 if cfg.half and device != "cpu" else 32
                        else:
                            options["half"] = cfg.half and device != "cpu"
                        messages.put({"type": "model", "message": f"Inference · {cfg.weights.name} · {device}", "names": model.names})
                    if stop.is_set():
                        break
                    if not inference_switch.is_set():
                        continue
                    with torch.inference_mode():
                        result = model.predict(frame, **options)[0]
                        annotated = result.plot(labels=cfg.labels, conf=cfg.labels)
                        if result.probs is not None:
                            records = [{"name": result.names[i], "class": i, "confidence": round(float(result.probs.data[i]), 5)} for i in result.probs.top5]
                        else:
                            records = result.summary()
                    inference_ms = float(result.speed.get("inference") or 0)
                    allocated = torch.cuda.memory_allocated(cfg.device) / 1024**2 if cfg.device != "cpu" else 0
                except Exception as exc:
                    if not cfg.live:
                        raise
                    model = None
                    inference_switch.clear()
                    enabled = False
                    if recorder and recorder.mode != "Original":
                        finish_recording("failed")
                    messages.put({"type": "inference_error", "message": str(exc) + "\nInference turned off. Camera preview and original recording remain available."})
            if describer and was_enabled and not enabled:
                describer.invalidate()
            was_enabled = enabled
            processed += 1
            now = time.perf_counter()
            source_seconds = (frame_index - 1) / fps if cfg.source_kind == "Video" else None
            if describer and enabled:
                describer.submit(frame, records, frame_index, source_seconds, captured_at,
                                 recorder.directory if recorder else None)
            if recorder:
                try:
                    recorder.write(frame, annotated, records, frame_index, source_seconds, captured_at, enabled)
                except Exception as exc:
                    finish_recording("failed")
                    messages.put({"type": "record_error", "message": str(exc)})
            now = time.perf_counter()
            completed_times.append(now)
            measured_fps = (len(completed_times)-1)/max(now-completed_times[0],.000001)
            if now - last_preview >= 1/20 or cfg.source_kind == "Image" or frame_index == total:
                h, w = annotated.shape[:2]
                ratio = min(1.0, 1280 / w, 900 / h)
                preview = cv2.resize(annotated, (max(1, int(w*ratio)), max(1, int(h*ratio)))) if ratio < 1 else annotated
                publish_latest(previews, {"rgb": cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), "records": records,
                    "comparison_bgr": cv2.resize(frame, (preview.shape[1], preview.shape[0])) if ratio < 1 else frame.copy(),
                    "original_size": [w, h], "task": cfg.task, "confidence": cfg.confidence, "source_seconds": source_seconds,
                    "count": len(records), "frame": frame_index, "total": total, "processed": processed,
                    "fps": measured_fps, "inference_ms": inference_ms, "gpu_mb": allocated,
                    "resolution": f"{w} × {h}", "inference": enabled, "recording": bool(recorder)})
                last_preview = now
            if cfg.source_kind == "Image":
                break
        if stop.is_set():
            outcome = "stopped"
    except Exception as exc:
        outcome = "failed"
        detail, trace = str(exc), traceback.format_exc()
        if cfg.source_kind == "Network stream" and cfg.source:
            detail, trace = detail.replace(cfg.source, "[stream]"), trace.replace(cfg.source, "[stream]")
        messages.put({"type": "error", "message": detail, "traceback": trace})
    finally:
        if describer:
            # Still images and EOF keep the last caption; Stop cancels promptly.
            describer.close(drain=outcome == "completed" and not stop.is_set())
        if outcome == "completed" and stop.is_set():
            outcome = "stopped"
        finish_recording(outcome)
        if camera:
            camera.release()
        elif capture:
            capture.release()
        messages.put({"type": "done", "outcome": outcome, "frames": processed})
