"""Validated, serializable settings shared by the GUI and inference process."""

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
TASKS = {
    "Object detection": "",
    "Instance segmentation": "-seg",
    "Pose estimation": "-pose",
    "Image classification": "-cls",
    "Oriented detection": "-obb",
}
SCALES = {"N · Nano": "n", "S · Small": "s", "M · Medium": "m",
          "L · Large": "l", "X · Extra large": "x"}
RECORD_MODES = ("Original", "With inference", "Both")


def model_name(task: str, scale: str) -> str:
    if task not in TASKS or scale not in SCALES:
        raise ValueError("Select a supported task and model size.")
    return f"yolo26{SCALES[scale]}{TASKS[task]}.pt"


def parse_classes(text: str) -> list[int] | None:
    if not text.strip():
        return None
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise ValueError("Class IDs must be comma-separated integers, for example 0,2,5.") from exc
    if any(value < 0 for value in values):
        raise ValueError("Class IDs cannot be negative.")
    return list(dict.fromkeys(values))


@dataclass(frozen=True)
class RunConfig:
    task: str = "Object detection"
    scale: str = "N · Nano"
    custom_model: str = ""
    source_kind: str = "Webcam"
    source: str = "0"
    inference_enabled: bool = True
    record_mode: str = "With inference"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    device: str = "cuda:0"
    image_size: int = 640
    confidence: float = 0.25
    max_detections: int = 100
    classes: str = ""
    half: bool = True
    cpu_threads: int = 2
    labels: bool = True
    frame_stride: int = 1
    save_output: bool = False
    output_dir: str = str(ROOT / "outputs")

    @property
    def weights(self) -> Path:
        return (Path(self.custom_model).expanduser().resolve() if self.custom_model
                else ROOT / "models" / model_name(self.task, self.scale))

    @property
    def live(self) -> bool:
        return self.source_kind in {"Webcam", "Pi camera (CSI)", "Network stream"}

    def validated_source(self) -> str | int:
        if self.inference_enabled:
            self.validate_inference()
        if not 1 <= self.frame_stride <= 30:
            raise ValueError("Frame stride must be between 1 and 30.")
        if not 1 <= self.cpu_threads <= 64:
            raise ValueError("CPU threads must be between 1 and 64.")
        if self.device != "cpu" and not (self.device.startswith("cuda:") and self.device[5:].isdigit()):
            raise ValueError("Select CPU or an available CUDA device.")
        if self.record_mode not in RECORD_MODES:
            raise ValueError("Choose Original, With inference, or Both for recording.")
        if not 1 <= self.camera_fps <= 120 or not 160 <= self.camera_width <= 3840 or not 120 <= self.camera_height <= 2160:
            raise ValueError("Camera size must be 160–3840 × 120–2160, with 1–120 FPS.")
        source = self.source.strip()
        if self.source_kind == 'Webcam':
            if source.isdigit():
                return int(source)
            if re.fullmatch(r'/dev/video\d+', source) or source.startswith(('/dev/v4l/by-id/', '/dev/v4l/by-path/')):
                return source
            raise ValueError('Select a camera, then click Refresh if it is missing.')
        if self.source_kind == "Pi camera (CSI)":
            if not source.isdigit():
                raise ValueError("Webcam index must be a non-negative integer, usually 0.")
            return int(source)
        if self.source_kind == "Network stream":
            url = urlsplit(source)
            if url.scheme not in {"rtsp", "rtsps", "http", "https", "rtmp"} or not url.hostname:
                raise ValueError("Enter an RTSP, HTTP(S), or RTMP media stream URL.")
            return source
        if self.source_kind not in {"Image", "Video"}:
            raise ValueError("Choose Image, Video, Webcam, Pi camera (CSI), or Network stream.")
        path = Path(source).expanduser()
        if not source or not path.is_file():
            raise ValueError("Choose an existing image or video file.")
        return str(path.resolve())

    def validate_inference(self):
        model_name(self.task, self.scale)
        if self.custom_model and (not self.weights.is_file() or self.weights.suffix != ".pt"):
            raise ValueError("Select an existing Ultralytics .pt model file.")
        if not 160 <= self.image_size <= 1536 or self.image_size % 32:
            raise ValueError("Image size must be a multiple of 32 between 160 and 1536.")
        if not 0.01 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0.01 and 1.00.")
        if not 1 <= self.max_detections <= 1000:
            raise ValueError("Maximum detections must be between 1 and 1000.")
        parse_classes(self.classes)
