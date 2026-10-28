# CNN + LLM Based Descriptor

A desktop visual descriptor combining a YOLO CNN detector with quantized Qwen3-VL 2B. It describes **only objects detected by YOLO**, with exact counts, bounding-box positions and confidence scores, then suggests a next step tied to a detected object.

A camera-first Python GUI for YOLO26 n / s / m / l / x, with NVIDIA GPU and CPU support, including Raspberry Pi and other 64-bit single-board computers.

## Open

Run **Run_CNN_LLM_Based_Descriptor.sh** with no arguments. On this PC it is in `~/Projects/CNN_LLM_Based_Descriptor`.

The launcher checks your installation. Select a device and click **Open CNN + LLM Based Descriptor**. **Installation options…** contains GUI setup/repair; it opens automatically when setup is needed. No setup flags are required.

Resize or maximize the app normally. It opens at 1600×1000 on larger displays and fits smaller screens. Buttons move onto additional rows when space is limited, status text wraps, and fonts adjust between readable small-window and large-window sizes. The camera preview scales to the available space while preserving its proportions. Settings and launcher windows are also resizable. A 1024×600 or larger desktop display is recommended; the main window can shrink to 760×520.

## Local vision descriptions

1. Install and start [Ollama](https://ollama.com/download) (0.12.7 or newer).
2. Open **Settings → Descriptions → Download model** (or run `python3 setup_descriptor.py` from this project). It downloads and verifies **qwen3-vl:2b**, a small **Q4_K_M (4-bit)** vision-language model, about **1.9 GB** on disk. Runtime RAM/VRAM usage is higher than download size. YOLO also needs memory; use Nano or CPU inference on a 4 GB GPU. The Qwen pipeline was tested with YOLO Nano on a 4 GB RTX 3050 Ti; structured descriptions took about 6–7 seconds in a warmed sample-image test (first loading can take longer), so it is not a per-frame real-time captioner. The model is cached by Ollama and is not committed to Git.
3. Open **Run_CNN_LLM_Based_Descriptor.sh**, select your source, and leave YOLO inference enabled. Descriptions are enabled by default. Use **Settings → Descriptions** to disable them or change the local model, server, sample interval, or timeout.
4. Read the **Detected objects and guidance** panel under the preview. It shows detected objects, a broad environment estimate, a suggested next step, and exact YOLO scores/positions. Its frame number (and source timestamp for video files) identifies the described frame; descriptions can lag behind live video. **Details** includes the full description, validation status, selected action and exact detection evidence.

The model receives the frame for context, but **cannot write free-form object descriptions**. It returns structured choices: `indoor`, `outdoor` or `unclear`, and one suggestion from a list tied to the supplied detection IDs. The application checks the complete answer and builds all displayed text from YOLO facts and fixed suggestion wording. Extra fields, unknown IDs, unsupported actions, prose and malformed answers are discarded; a YOLO-only summary is shown instead and the next frame can try again. Unchecked model text is never displayed or saved in description records.

Suggestions are general scene guidance: inspect a detected object, allow space if moving nearby, confirm a low-score detection, inspect before handling a detected knife/scissors, or check the actual state of a detected signal. These suggestions do not assert motion, signal color, distance, danger, or that a route is clear. For example, a person detected at the left may prompt “If moving nearby, leave space around the detected person.” A visible object absent from YOLO's supplied detections cannot appear in the object summary or suggestion. Indoor/outdoor context is separately marked as a **model estimate** and can be wrong.

The Qwen3-VL compatibility prefill avoids the forced reasoning in older Ollama builds so the bounded output budget is used for these structured choices.

The default samples at most once every three seconds, with a 220-token output limit and a 180-second request timeout to allow the first model load. One background request and one latest pending frame keep memory bounded and video preview responsive. Cancel download stops after the current network read; downloading again resumes cached progress. Ollama chooses its GPU/CPU placement independently of YOLO. The model receives the original frame resized to at most 640 px, plus up to 10 predictions ordered by confidence: IDs, labels, scores, normalized boxes, and spatial positions. Exact pixel boxes remain in the saved evidence. Only valid boxes at or above the selected detection threshold qualify. Axis-aligned extents represent oriented boxes. Extra predictions and objects dropped to fit the context window are counted as omitted and cannot be suggestion targets. Counts cover listed predictions only. With no detections, the app skips the model and says no objects were detected; it does not call the scene empty. Classification results are shown as class candidates, with no object counts, positions or object-specific advice, and no vision-model call.

Confidence percentages are **YOLO detector scores**, not measured accuracy or confidence in the environment estimate. Detections below 50% are marked tentative and can only receive a verification suggestion. YOLO can still miss or misclassify objects; the descriptor preserves its predictions rather than substituting the vision model's guesses. Suggestions are conditional guidance for reviewing the scene, not movement commands. No cloud endpoint is used: the server must be loopback, proxy forwarding and redirects are disabled, and cloud-only models are rejected. Alternative local models must report vision support and quantized weights and support structured JSON output.

Descriptions are generated only when YOLO inference is on. A missing model, unavailable server, invalid model or timeout disables descriptions for that run and displays recovery instructions; detection and recording continue. Reopen the input after fixing the problem. Still images and normal video EOF wait for pending descriptions. **Stop** cancels pending work without waiting for a model response; an already submitted request can finish inside Ollama, but its caption is discarded. It never stops a shared Ollama service.

When recording or saving an image, sampled descriptions are written to `descriptions.jsonl` beside `predictions.jsonl`, with their original frame number, source time, model name, generation time, frame dimensions and exact evidence. A caption can finish after its recording is stopped, and still belongs to that recording's directory. Manual Stop can discard unfinished captions. Captions are displayed in the panel and saved as metadata; they are not burned into video frames.

Source: [Qwen3-VL 2B model and quantization](https://ollama.com/library/qwen3-vl:2b), [Ollama vision API](https://docs.ollama.com/capabilities/vision), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

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
- **Descriptions:** local quantized vision model, loopback Ollama server, sampling interval and timeout.
- **Saving:** optional default save folder and **Record from start / save image result** for automatic export, including short files that might finish before you can click Record. Automatic video recording asks for a filename when opening the source. Cancelling starts preview without recording.

Change these settings before opening the source. A camera may choose a supported capture size or FPS different from the request. Model checkpoints download on first use; Nano, Small, Medium, and Large detection checkpoints are cached on this PC.

Images, video files, direct network streams, and Raspberry Pi CSI cameras are also available in the input selector. **Video** means an existing input video to play, and only that input mode needs an existing video file. Use **Webcam** for live camera viewing or recording. Video files are processed as fast as possible.

## Output and timing

Videos are saved at your selected MP4 filename. In **Both** mode, choosing `clip.mp4` saves annotations there and clean frames in `clip_original.mp4`. A companion `clip_details` folder contains `predictions.jsonl`, `settings.json`, and `run.json`. Image exports use timestamped folders with `.jpg` files. Videos are silent. Per-frame JSON records say whether inference was enabled and contain predictions when available. Network stream URLs are omitted from saved settings.

Live capture continuously reads the newest camera frame to avoid old frames accumulating. Recording uses the frames processed by the app. When inference or encoding is slower than capture, some captured frames are omitted and the previous processed frame is repeated in the recording to preserve elapsed playback time. Original-only recording with inference off uses the lighter camera-only path. Requested camera FPS is not a guaranteed processing rate. The displayed FPS is the measured processing rate over recent frames.

Saved videos use the capture resolution. Preview snapshots are limited to 1280×900. File-video frame skipping adjusts playback FPS accordingly. Stop normally finalizes MP4 files; a blocked worker is force-stopped after eight seconds, which can leave the current clip incomplete.

## Raspberry Pi / SBC installation

Copy `Run_CNN_LLM_Based_Descriptor.sh`, `launcher.py`, `main.py`, `setup_descriptor.py`, `requirements.txt`, and the `cnn_llm_based_descriptor/` folder into a `CNN_LLM_Based_Descriptor` folder on a **64-bit Linux desktop with Python 3.10+**, then open **Run_CNN_LLM_Based_Descriptor.sh**. In **Installation options…**, select **CPU · Raspberry Pi / SBC / PC** and click **Set up / repair**.

**Install Linux prerequisites…** installs Tk, venv, pip, and graphics libraries on apt-based Linux systems using the OS authentication dialog. If Tk is missing before the launcher can open, it offers a graphical bootstrap on systems with Zenity and PolicyKit; otherwise install Python Tk through the OS software manager. System package installation must finish before the window closes.

For a Raspberry Pi CSI camera, enable the CSI-support checkbox before installing Linux prerequisites. In the app, choose **Pi camera (CSI)** and index **0**. ARM environments expose system Picamera2/libcamera packages. Other boards can use compatible USB cameras. Pi/ARM hardware and CSI cameras have not been physically tested; the CPU path has been tested on this PC.

Use **Settings → Camera → Raspberry Pi / SBC preset** to select Nano, CPU, 320-pixel inference, two CPU threads, and frame stride 2. Larger models need more memory and may be too slow on small boards. No board-specific NPU acceleration is included.

Copy the `models/` folder separately for offline use. Do not copy `.venv/`, `.environments/`, or `launcher_settings.json` between computers.

## Installation management

GUI setup offers existing-package reuse, CPU-only PyTorch, or NVIDIA CUDA packages. CUDA 13.0 matches this PC; CUDA 12.8 is also offered. Driver compatibility is described in [PyTorch's installation guide](https://pytorch.org/get-started/locally/). Setup does not change GPU drivers.

New environments are verified before activation; failed or cancelled installs leave the working environment available. The active environment is recorded in `launcher_settings.json`, with the original `.venv/` also supported. Repairs retain prior working environments and can require additional disk space.

## Verification

The final combined version passes 19 automated tests, compact-window GUI checks, and a real CUDA YOLO → quantized Qwen → saved-description test on the bundled bus image. GitHub Actions runs the automated checks on pushes and pull requests.

The descriptor update passes 17 automated tests, Python syntax checks and launcher shell checks. The updated GUI was checked at 760×520 and 1200×850, including the description panel and Settings. The original detector GUI had broader layout checks before this update.

Earlier detector-only camera checks on this PC passed live preview, CUDA inference, both MP4 tracks, stopping, and reopening. A simulated USB disconnect verified recovery of the same camera and a playable interrupted recording. Earlier checks covered CPU image/video inference, file-free preview, recording filename selection, cancellation, protection of existing files, and launcher setup. Raspberry Pi/ARM camera hardware remains untested.

Runtime libraries may recreate caches when the app is used. Git excludes local environments, model weights, settings, logs, and recordings.

## References

[YOLO26 models](https://docs.ultralytics.com/models/yolo26/) · [Prediction API](https://docs.ultralytics.com/modes/predict/) · [Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) · [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)

[Linux camera capabilities](https://cdn.kernel.org/doc/html/latest/userspace-api/media/v4l/vidioc-querycap.html) · [OpenCV camera backends](https://docs.opencv.org/4.13.0/d8/dfe/classcv_1_1VideoCapture.html)

Ultralytics code/models have their own AGPL-3.0 / enterprise terms. Dependencies retain their respective licenses.

## Development checks

Run `python3 -m unittest discover -s tests -v` after installing `requirements.txt`. These checks cover evidence conversion, frame/image requests, error isolation, stale-result cancellation, bounded queuing, saving and still-image inference integration; they do not download a model.

The application package is `cnn_llm_based_descriptor`. The GitHub repository remains available at the following origin:

```bash
git remote set-url origin https://github.com/denizkarya1999/CNN_LLM_Based_Descriptor.git
```
