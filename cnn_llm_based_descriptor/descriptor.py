"""Bounded asynchronous frame scheduling, shared by all descriptor approaches."""
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time

from .algorithms.factory import create_descriptor_client
from .methods import method_details
from .measurement import measure_description

class FrameDescriber:
    """One request in flight and one latest pending frame; never block YOLO."""
    def __init__(self, cfg, messages, stop, inference_switch, client=None):
        self.cfg, self.messages, self.stop, self.inference_switch = cfg, messages, stop, inference_switch
        self.client = client if client is not None else create_descriptor_client(cfg)
        self.pending = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.busy = threading.Event()
        self.lock = threading.Lock()
        self.generation = 0
        self.failed = False
        self.next_at = 0.0
        self.thread = threading.Thread(target=self._run, name="frame-describer", daemon=True)
        self.thread.start()

    def submit(self, frame, records, frame_index, source_seconds, captured_at, directory=None):
        now = time.monotonic()
        if self.failed or self.closed.is_set() or now < self.next_at:
            return False
        self.next_at = now + self.cfg.describe_interval
        job = {"frame": frame_index, "source_seconds": source_seconds,
               "captured_at_monotonic": captured_at,
               "submitted_at": datetime.now(timezone.utc).isoformat(),
               "directory": str(directory) if directory else None,
               "image": frame.copy(), "records": records, "generation": self.generation}
        with self.lock:
            self._clear_pending()
            self.pending.put_nowait(job)
        return True

    def _clear_pending(self):
        try:
            self.pending.get_nowait()
        except queue.Empty:
            pass

    def invalidate(self):
        with self.lock:
            self.generation += 1
            self.next_at = 0.0
            self._clear_pending()

    def _run(self):
        while not self.closed.is_set() and not self.stop.is_set():
            with self.lock:
                try:
                    job = self.pending.get_nowait()
                    self.busy.set()
                except queue.Empty:
                    job = None
            if job is None:
                self.closed.wait(.03)
                continue
            try:
                if self.failed or not self.inference_switch.is_set():
                    continue
                result = measure_description(self.client, job["image"], job["records"])
                packet = {key: value for key, value in job.items() if key not in {"image", "records", "generation", "directory"}}
                packet.update(result)
                packet.update(method_details(self.cfg), type="description")
                with self.lock:
                    if (job["generation"] != self.generation or self.closed.is_set()
                            or self.stop.is_set() or not self.inference_switch.is_set()):
                        continue
                    self.messages.put(packet)
                    if job["directory"]:
                        try:
                            with (Path(job["directory"]) / "descriptions.jsonl").open("a", encoding="utf-8") as output:
                                output.write(json.dumps(packet, ensure_ascii=False) + "\n")
                        except OSError as exc:
                            self.messages.put({"type": "description_save_error", "message": f"Description shown but could not be saved: {exc}"})
            except Exception as exc:
                with self.lock:
                    if job["generation"] == self.generation and not self.closed.is_set() and not self.stop.is_set():
                        self.failed = True
                        self._clear_pending()
                        self.messages.put({"type": "description_error", "message": str(exc)})
            finally:
                self.busy.clear()

    def close(self, drain=False):
        if drain:
            while not self.stop.is_set():
                with self.lock:
                    finished = self.pending.empty() and not self.busy.is_set()
                if finished:
                    break
                self.stop.wait(.05)
        self.closed.set()
        self.invalidate()
        self.thread.join(.2)
