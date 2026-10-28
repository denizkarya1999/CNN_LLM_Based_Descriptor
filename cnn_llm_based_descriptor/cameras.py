"""Find real Linux capture devices without starting a camera stream."""
import os
from pathlib import Path
import struct
import sys


def capture_device(path):
    """QUERYCAP distinguishes camera frames from UVC metadata/codec nodes."""
    import fcntl
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        data = bytearray(104)  # struct v4l2_capability from linux/videodev2.h
        fcntl.ioctl(fd, 0x80685600, data)  # VIDIOC_QUERYCAP
    finally:
        os.close(fd)
    _, card, _, _, caps, device_caps = struct.unpack_from('16s32s32sIII', data)
    flags = device_caps if caps & 0x80000000 else caps
    if not flags & 0x00001001 or flags & 0x0000c000:
        return None
    return card.split(b'\0')[0].decode(errors='replace') or Path(path).name


def discover_cameras():
    """Return selectable cameras and access errors; never guess indices 0–9."""
    if not sys.platform.startswith('linux'):
        return [], []
    aliases = {}
    for folder in ('/dev/v4l/by-path', '/dev/v4l/by-id'):
        for link in sorted(Path(folder).glob('*')):
            aliases[str(link.resolve())] = str(link)
    devices, errors = [], []
    nodes = [p for p in Path('/dev').glob('video*') if p.name[5:].isdigit()]
    for node in sorted(nodes, key=lambda p: int(p.name[5:])):
        try:
            name = capture_device(node)
            if name:
                devices.append({'path': aliases.get(str(node), str(node)), 'node': str(node),
                                'name': name, 'label': f'{name} · {node.name}'})
        except PermissionError:
            errors.append(f'Camera access denied for {node}. Check your camera permissions.')
        except OSError:
            # Devices may disappear while the list is being refreshed.
            continue
    return devices, errors


def linux_camera_path(source):
    text = str(source).strip()
    return Path('/dev') / f'video{text}' if text.isdigit() else Path(text)


def open_webcam(cfg, source):
    import cv2
    target = source
    if sys.platform.startswith('linux'):
        target = linux_camera_path(source)
        try:
            name = capture_device(target)
        except FileNotFoundError as exc:
            raise RuntimeError(f'Camera {target} is disconnected. Reconnect it, click Refresh, then Open camera.') from exc
        except PermissionError as exc:
            raise RuntimeError(f'Camera access denied for {target}. Check your camera permissions.') from exc
        except OSError as exc:
            raise RuntimeError(f'Camera {target} is unavailable. Reconnect it and click Refresh.') from exc
        if name is None:
            raise RuntimeError(f'{target} does not supply camera pictures. Click Refresh and select a camera by name.')
        # An explicit backend avoids FFmpeg/obsensor trying a missing V4L2 node.
        capture = cv2.VideoCapture(str(target), cv2.CAP_V4L2)
    else:
        capture = cv2.VideoCapture(target)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f'Cannot open camera {target}. Close other camera apps, reconnect the camera, and click Refresh.')
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
        capture.set(cv2.CAP_PROP_FPS, cfg.camera_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        capture.release()
        raise
    return capture
