"""Original/annotated recording with wall-clock timing for live sources."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path


def recording_destinations(filename, mode):
    """Keep the selected filename and any companion files together."""
    path = Path(filename).expanduser().resolve()
    if path.suffix.lower() != '.mp4':
        raise ValueError('Choose an MP4 filename ending in .mp4.')
    paths = {'original' if mode == 'Original' else 'annotated': path}
    if mode == 'Both':
        paths['original'] = path.with_name(path.stem + '_original.mp4')
    return path.with_name(path.stem + '_details'), paths


class VideoTrack:
    def __init__(self, path, fps):
        self.path, self.fps = str(path), fps
        self.writer = self.previous = None
        self.size = None
        self.index = -1

    def write(self, frame, seconds=None):
        import cv2
        h, w = frame.shape[:2]
        if self.writer is None:
            self.size = (w, h)
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self.writer.isOpened():
                self.writer.release()
                raise OSError("Could not open the MP4 writer. Check output permissions and codec support.")
        if (w, h) != self.size:
            frame = cv2.resize(frame, self.size)
        target = self.index + 1 if seconds is None else int(max(0, seconds) * self.fps)
        if target <= self.index:
            return
        # Repeat the previous processed frame across inference gaps so a live
        # recording keeps its elapsed duration instead of playing at fast speed.
        while self.previous is not None and self.index + 1 < target:
            self.writer.write(self.previous)
            self.index += 1
        self.writer.write(frame)
        self.previous = frame.copy()
        self.index += 1

    def close(self, seconds=None):
        if self.writer is not None:
            try:
                if seconds is not None and self.previous is not None:
                    self.write(self.previous, seconds)
            finally:
                self.writer.release()


class RunRecorder:
    def __init__(self, cfg, fps, mode, started, output_path=None):
        self.cfg, self.mode, self.started = cfg, mode, started
        self.count = 0
        if output_path:
            self.directory, paths = recording_destinations(output_path, mode)
        else:
            self.directory = Path(cfg.output_dir).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            names = ["original"] if mode == "Original" else ["annotated"] if mode == "With inference" else ["original", "annotated"]
            paths = {name: self.directory / f'{name}.mp4' for name in names}
        self.directory.mkdir(parents=True, exist_ok=False)
        settings = asdict(cfg)
        settings["record_mode"] = mode
        if cfg.source_kind == "Network stream":
            settings["source"] = "[network stream URL omitted]"
        reserved = []
        try:
            if output_path:
                for path in paths.values():
                    # Reserve every destination before OpenCV can truncate a file.
                    with path.open('xb'):
                        pass
                    reserved.append(path)
            (self.directory / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
            self.records = (self.directory / "predictions.jsonl").open("w", encoding="utf-8")
        except Exception:
            for path in reserved:
                path.unlink(missing_ok=True)
            for name in ('settings.json', 'predictions.jsonl'):
                (self.directory / name).unlink(missing_ok=True)
            self.directory.rmdir()
            raise
        self.tracks = {name: VideoTrack(path, fps) for name, path in paths.items()}

    def write(self, original, annotated, records, frame_index, source_seconds, now, inference):
        import cv2
        seconds = max(0, now - self.started) if self.cfg.live else None
        for name, track in self.tracks.items():
            frame = original if name == "original" else annotated
            if self.cfg.source_kind == "Image":
                if not cv2.imwrite(str(self.directory / f"{name}.jpg"), frame):
                    raise OSError("Could not save image. Check output permissions and disk space.")
            else:
                track.write(frame, seconds)
        self.records.write(json.dumps({"frame": frame_index, "source_seconds": source_seconds,
            "recording_seconds": seconds, "inference": inference, "predictions": records}) + "\n")
        self.records.flush()
        self.count += 1

    def close(self, status, now):
        seconds = max(0, now - self.started) if self.cfg.live else None
        try:
            for track in self.tracks.values():
                track.close(seconds)
        finally:
            self.records.close()
        (self.directory / "run.json").write_text(json.dumps({"status": status, "samples": self.count,
            "frames": self.count, "record_mode": self.mode, "duration_seconds": seconds,
            "video_files": {name: track.path for name, track in self.tracks.items()},
            "video_frames": {name: max(0, track.index + 1) for name, track in self.tracks.items()}}, indent=2), encoding="utf-8")
