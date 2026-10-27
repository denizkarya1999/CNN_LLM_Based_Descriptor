# Generic YOLO

A camera-first Python GUI for YOLO26 n / s / m / l / x, with NVIDIA GPU and CPU support, including Raspberry Pi and other 64-bit single-board computers.

## Open

Run **Run_Generic_YOLO.sh** with no arguments. On this PC it is in `~/Projects/Generic_YOLO`.

The launcher checks your installation. Select a device and click **Open Generic YOLO**. **Installation options…** contains GUI setup/repair; it opens automatically when setup is needed. No setup flags are required.

Resize or maximize the app normally. It opens at 1600×1000 on larger displays and fits smaller screens. Buttons move onto additional rows when space is limited, status text wraps, and fonts adjust between readable small-window and large-window sizes. The camera preview scales to the available space while preserving its proportions. Settings and launcher windows are also resizable. A 1024×600 or larger desktop display is recommended; the main window can shrink to 760×520.

## Live camera and recording

1. Choose **Webcam**, select your camera by name, and click **Open camera**. On Linux, **Refresh** updates the list after plugging in a camera. Other operating systems use a camera index, usually **0**. Live preview and inference do not need a video file or save location. An empty or missing saved file input resets to Webcam when the app opens.
2. Use **Enable inference** to turn YOLO on or off while the camera stays open. Turn it off before opening the camera for camera-only operation without loading a model.
3. Select a recording mode, then click **Record** and choose an MP4 filename. Cancel the dialog to keep previewing without recording:

   | Mode | Saved video |
   | --- | --- |
   | Original | Clean camera frames without overlays, whether inference is on or off |
   | With inference | Annotated video with model results |
   | Both | Annotated video at the chosen filename, plus a separate `_original.mp4` video |

4. Click **Stop recording** to finish the clip. The camera continues previewing, and you can start another clip.
5. Click **Stop** (or press Esc) to finish any recording and close the camera.

Overlay recording requires inference. Turning inference off during an annotated/Both recording finishes that clip; original-only recording can continue across inference changes. Model loading and inference finish their current operation before changes take effect. If inference fails during a camera session, it is turned off and the original camera preview remains available.

**Snapshot** saves the current preview. **Outputs** opens the folder containing the latest clip. **Details** shows predictions, device information, and the activity log. Existing recording files are protected; choose a new filename for each clip.

Linux camera discovery lists video capture devices and excludes metadata/codec nodes. Stable device links are used when available, so a camera can be found even if its `/dev/videoN` number changes. If a webcam disconnects while running, the app finalizes the current recording and tries to reconnect to the same camera for up to eight seconds. Recording stays stopped after recovery; click **Record** to start a new clip. If recovery fails, check the connection or privacy switch, close other camera apps, click **Refresh**, and open the camera again. The error stays visible in the main window. USB hardware/driver disconnects can still interrupt capture.

## Settings

The main screen contains input, model size, device, inference, and recording controls. **Settings** holds the less frequently used options:

- **Model:** detection, instance segmentation, pose, classification, or oriented detection; inference resolution, confidence, class filter, custom weights, FP16, and labels.
- **Camera:** requested resolution/FPS, inference frame stride, CPU threads, and the **Raspberry Pi / SBC preset**.
- **Saving:** optional default save folder and **Record from start / save image result** for automatic export, including short files that might finish before you can click Record. Automatic video recording asks for a filename when opening the source. Cancelling starts preview without recording.

Change these settings before opening the source. A camera may choose a supported capture size or FPS different from the request. Model checkpoints download on first use; Nano, Small, Medium, and Large detection checkpoints are cached on this PC.

Images, video files, direct network streams, and Raspberry Pi CSI cameras are also available in the input selector. **Video** means an existing input video to play, and only that input mode needs an existing video file. Use **Webcam** for live camera viewing or recording. Video files are processed as fast as possible.

## Output and timing

Videos are saved at your selected MP4 filename. In **Both** mode, choosing `clip.mp4` saves annotations there and clean frames in `clip_original.mp4`. A companion `clip_details` folder contains `predictions.jsonl`, `settings.json`, and `run.json`. Image exports use timestamped folders with `.jpg` files. Videos are silent. Per-frame JSON records say whether inference was enabled and contain predictions when available. Network stream URLs are omitted from saved settings.

Live capture continuously reads the newest camera frame to avoid old frames accumulating. Recording uses the frames processed by the app. When inference or encoding is slower than capture, some captured frames are omitted and the previous processed frame is repeated in the recording to preserve elapsed playback time. Original-only recording with inference off uses the lighter camera-only path. Requested camera FPS is not a guaranteed processing rate. The displayed FPS is the measured processing rate over recent frames.

Saved videos use the capture resolution. Preview snapshots are limited to 1280×900. File-video frame skipping adjusts playback FPS accordingly. Stop normally finalizes MP4 files; a blocked worker is force-stopped after eight seconds, which can leave the current clip incomplete.

## Raspberry Pi / SBC installation

Copy `Run_Generic_YOLO.sh`, `launcher.py`, `main.py`, `requirements.txt`, and the `generic_yolo/` folder into a `Generic_YOLO` folder on a **64-bit Linux desktop with Python 3.10+**, then open **Run_Generic_YOLO.sh**. In **Installation options…**, select **CPU · Raspberry Pi / SBC / PC** and click **Set up / repair**.

**Install Linux prerequisites…** installs Tk, venv, pip, and graphics libraries on apt-based Linux systems using the OS authentication dialog. If Tk is missing before the launcher can open, it offers a graphical bootstrap on systems with Zenity and PolicyKit; otherwise install Python Tk through the OS software manager. System package installation must finish before the window closes.

For a Raspberry Pi CSI camera, enable the CSI-support checkbox before installing Linux prerequisites. In the app, choose **Pi camera (CSI)** and index **0**. ARM environments expose system Picamera2/libcamera packages. Other boards can use compatible USB cameras. Pi/ARM hardware and CSI cameras have not been physically tested; the CPU path has been tested on this PC.

Use **Settings → Camera → Raspberry Pi / SBC preset** to select Nano, CPU, 320-pixel inference, two CPU threads, and frame stride 2. Larger models need more memory and may be too slow on small boards. No board-specific NPU acceleration is included.

Copy the `models/` folder separately for offline use. Do not copy `.venv/`, `.environments/`, or `launcher_settings.json` between computers.

## Installation management

GUI setup offers existing-package reuse, CPU-only PyTorch, or NVIDIA CUDA packages. CUDA 13.0 matches this PC; CUDA 12.8 is also offered. Driver compatibility is described in [PyTorch's installation guide](https://pytorch.org/get-started/locally/). Setup does not change GPU drivers.

New environments are verified before activation; failed or cancelled installs leave the working environment available. The active environment is recorded in `launcher_settings.json`, with the original `.venv/` also supported. Repairs retain prior working environments and can require additional disk space.

## Verification

The responsive GUI was checked from 760×520 through 3800×2080, including every input mode, font scaling, Settings tabs, and compact/expanded launcher layouts. Visible controls were checked for clipping and overlap. Syntax checks also passed.

Camera checks on this PC passed live preview, CUDA inference, both MP4 tracks, stopping, and reopening. A simulated USB disconnect verified recovery of the same camera and a playable interrupted recording. Earlier checks covered CPU image/video inference, file-free preview, recording filename selection, cancellation, protection of existing files, and launcher setup. Raspberry Pi/ARM camera hardware remains untested.

Temporary verification scripts and generated caches were removed after checking. Runtime libraries may recreate caches when the app is used. Git excludes local environments, model weights, settings, logs, and recordings.

## References

[YOLO26 models](https://docs.ultralytics.com/models/yolo26/) · [Prediction API](https://docs.ultralytics.com/modes/predict/) · [Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) · [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)

[Linux camera capabilities](https://cdn.kernel.org/doc/html/latest/userspace-api/media/v4l/vidioc-querycap.html) · [OpenCV camera backends](https://docs.opencv.org/4.13.0/d8/dfe/classcv_1_1VideoCapture.html)

Ultralytics code/models have their own AGPL-3.0 / enterprise terms. Dependencies retain their respective licenses.
