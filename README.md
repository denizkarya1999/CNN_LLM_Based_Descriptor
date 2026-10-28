<img src="assets/icon-64.png" alt="CNN + LLM Based Descriptor icon" width="64" height="64">

# CNN + LLM Based Descriptor

A desktop visual descriptor combining a YOLO CNN detector with either quantized Qwen3-VL 2B or a **CIS 579 probability, logic, and search** alternative. It describes **only objects detected by YOLO**, with exact counts, bounding-box positions and confidence scores, then suggests a next step tied to a detected object.

A camera-first Python GUI for YOLO26 n / s / m / l / x, with NVIDIA GPU and CPU support, including Raspberry Pi and other 64-bit single-board computers.

## Screenshots

The main app shows the live preview and scene description together, including detections, environment, decision, and reason. Descriptions identify their sampled frame and can lag behind the live preview.

![Main app with live YOLO detections and scene description](Screenshots/Main%20App.png)

The launcher provides device selection, installation status, and setup controls.

![Launcher with device selection and setup controls](Screenshots/Launcher.png)

## Open

Run **Run_CNN_LLM_Based_Descriptor.sh** with no arguments. On this PC it is in `~/Projects/CNN_LLM_Based_Descriptor`.

The launcher checks your installation. Select a device and click **Open app**. Launch, check, cancel, repair and Linux prerequisite buttons remain visible in a fixed bottom bar. Installation options and the activity log scroll independently. No setup flags are required.

The app opens at 1440×900 on larger displays, with the camera preview and scene description side by side. Preview and Description remain visible together at every supported size: side by side on wide or short windows, and stacked on narrow, taller windows. The description scrolls independently. Open, Stop, Record and Snapshot stay in a fixed bottom bar. **Decision:** is prominent, with amber **Take Action** and teal **No Action** highlights. Settings pages scroll, including every model-download button. The app supports windows down to 760×520; the launcher supports 640×480 and does not shrink itself after checking dependencies.

## Choose how scenes are described

Open **Settings → Descriptions → Description method** and select **LLM** or **CIS 579** before opening the input. Both methods share the same Preview and Scene understanding GUI, decision highlights, and recording metadata. Existing installations keep LLM selected.

- **LLM:** Qwen uses the image to estimate what the setting looks like: a home/living area, ballroom/event hall, kitchen, bedroom, bathroom, office, classroom, restaurant, shop, street, park, sports area, transport station, or vehicle interior. It can say the setting is unclear. These are unconfirmed environment estimates, separate from YOLO object facts.
- **CIS 579:** works without Ollama or an LLM download. It applies conditional probability and exact enumeration to detected class evidence, implication rules to identify review goals, and **uniform-cost search (UCS)** to suggest a review sequence. The image pixels are not analyzed by this alternative; only YOLO classes, scores, and boxes are used.

### CIS 579 probability and search

The implementation connects the conditional-probability/inference-by-enumeration material from Lecture 6 with rule-based reasoning and state-space search. It is an explicit teaching model: the priors, likelihoods, review rules, and search costs are hand-authored, not learned or calibrated from a scene dataset.

For scene hypothesis `S` and distinct observed classes `E`, it enumerates and normalizes:

`P(S | E) = P(S) × product(P(class observed | S)) / sum_over_scenes(P(scene) × product(P(class observed | scene)))`

This defines a small joint model with classes assumed conditionally independent given the scene. Missing classes are unobserved and marginalized; failing to detect an object is not evidence of absence. A YOLO score of at least 50% admits a class as a cue; the score itself is **not** used as a calibrated likelihood. Duplicate detections do not multiply the class evidence. Unsupported labels provide no scene evidence. The editable table is [`scene_model.json`](cnn_llm_based_descriptor/scene_model.json).

The GUI shows up to three competing scene hypotheses and labels their probabilities **illustrative, not measured accuracy**. A setting is selected only with at least two supported cues, a top posterior of at least 55%, and a lead of at least 15 percentage points; otherwise it stays unclear. For example, a couch, TV, and remote favor a home/living area; a person alone establishes no place type. Tables/chairs can fit both a restaurant and an event hall. Validation on labeled scenes is needed before interpreting these numbers as real-world accuracy.

Rules create review goals for low-score detections, detected signals whose states need checking, and knives/scissors if they will be handled. Other object presence alone yields **No Action**. UCS uses a priority queue and a state consisting of the targets included in the plan plus the last target. A transition adds an allowed, detection-linked suggestion. The goal covers the selected review targets. Cost is illustrative review effort plus the normalized Manhattan shift between image-box centers. This is a plan for reviewing an image, **not a navigation route or an executed action**. It does not infer physical distance, motion, or danger.

Search is bounded to eight review targets from the listed detections; any omitted targets are reported. The first suggestion becomes **Take Action**, with its detected object and explanation. **Details** and `descriptions.jsonl` include the full posterior, admitted cues, rule identifiers, search steps, cost, expanded-state count, and omitted-target count. No probability model runs for classification-only input or when there are no valid detections.

Algorithm references: [conditional probability and inference by enumeration](https://inst.eecs.berkeley.edu/~cs188/textbook/bayes-nets/inference.html), [uniform-cost search](https://inst.eecs.berkeley.edu/~cs188/textbook/search/uninformed.html#133-uniform-cost-search).

## Local vision descriptions

1. Install [Ollama](https://ollama.com/download) (0.12.7 or newer). The app starts its own local model service when needed.
2. Open **Settings → Descriptions → Download model** (or run `python3 setup_descriptor.py` from this project). It downloads and verifies **qwen3-vl:2b**, a small **Q4_K_M (4-bit)** vision-language model, about **1.9 GB** on disk. Runtime RAM/VRAM usage is higher than download size. YOLO also needs memory; use Nano or CPU inference on a 4 GB GPU. The Qwen pipeline was tested with YOLO Nano on a 4 GB RTX 3050 Ti; structured descriptions took about 6–7 seconds in a warmed sample-image test (first loading can take longer), so it is not a per-frame real-time captioner. LLM weights are stored in the project’s `Models/ollama/` folder; YOLO weights are stored directly in `Models/`. Weights are excluded from Git.
3. Open **Run_CNN_LLM_Based_Descriptor.sh**, select your source, and leave YOLO inference enabled. Descriptions are enabled by default. Use **Settings → Descriptions** to disable them or change the local model, server, sample interval, or timeout.
4. Read the **Scene understanding** panel beside or below the preview. It shows **Detections** (YOLO objects, counts, scores and positions), **Environment** (what kind of place it looks like, marked as an estimate), **Decision** (either **Take Action** or **No Action**), and **Reason** (the evidence behind that recommendation). Its frame number (and source timestamp for video files) identifies the described frame; descriptions can lag behind live video. **Details** includes the full description, validation status, action decision, reason and exact detection evidence.

The model receives the frame for context, but **cannot write free-form object descriptions**. It returns three structured choices: an `environment` (`indoor`, `outdoor`, or `unclear`), a `setting` from the supported place categories, and `next_step`, either `no_action` or one suggestion from a list tied to the supplied detection IDs. Object presence alone does not force an action. The application checks the complete answer and builds all displayed text from YOLO facts and fixed suggestion/reason wording. Reasons for an action identify its detected object, position and exact confidence score, then explain the selected suggestion. Extra fields, unknown IDs, unsupported actions/settings, contradictory indoor/outdoor choices, prose and malformed answers are discarded; a YOLO-only summary is shown instead and the next frame can try again. Unchecked model text is never displayed or saved in description records.

**No Action** means no specific action is recommended from the supplied evidence; it does not declare the scene safe. An empty detection list, classification-only input or an invalid model answer also shows **No Action**, with **Reason** explaining why a recommendation is unavailable. A valid model answer can select **No Action** even when objects are detected.

Suggestions are general scene guidance: inspect a detected object, allow space if moving nearby, confirm a low-score detection, inspect before handling a detected knife/scissors, or check the actual state of a detected signal. These suggestions do not assert motion, signal color, distance, danger, or that a route is clear. For example, a person detected at the left may prompt “If moving nearby, leave space around the detected person.” A visible object absent from YOLO's supplied detections cannot appear in the object summary or suggestion. The place type and indoor/outdoor context are separately marked as **model estimates** and can be wrong. The setting label cannot introduce extra object descriptions.

The Qwen3-VL compatibility prefill avoids the forced reasoning in older Ollama builds so the bounded output budget is used for these structured choices.

The default samples at most once every three seconds, with a 220-token output limit and a 180-second request timeout to allow the first model load. One background request and one latest pending frame keep memory bounded and video preview responsive. Cancel download stops after the current network read; downloading again resumes cached progress. Ollama chooses its GPU/CPU placement independently of YOLO. The model receives the original frame resized to at most 640 px, plus up to 10 predictions ordered by confidence: IDs, labels, scores, normalized boxes, and spatial positions. Exact pixel boxes remain in the saved evidence. Only valid boxes at or above the selected detection threshold qualify. Axis-aligned extents represent oriented boxes. Extra predictions and objects dropped to fit the context window are counted as omitted and cannot be suggestion targets. Counts cover listed predictions only. With no detections, the app skips the model and says no objects were detected; it does not call the scene empty. Classification results are shown as class candidates, with no object counts, positions or object-specific advice, and no vision-model call.

Confidence percentages are **YOLO detector scores**, not measured accuracy or confidence in the environment estimate. Detections below 50% are marked tentative and can only receive a verification suggestion. YOLO can still miss or misclassify objects; the descriptor preserves its predictions rather than substituting the vision model's guesses. Suggestions are conditional guidance for reviewing the scene, not movement commands. No cloud endpoint is used: the server must be loopback, proxy forwarding and redirects are disabled, and cloud-only models are rejected. Alternative local models must report vision support and quantized weights and support structured JSON output.

Descriptions are generated only when YOLO inference is on. A missing model, failed server startup, invalid model or timeout disables descriptions for that run and displays recovery instructions; detection and recording continue. Reopen the input after fixing the problem. Still images and normal video EOF wait for pending descriptions. **Stop** cancels pending work without waiting for a model response; an already submitted request can finish inside Ollama, but its caption is discarded. The dedicated project service uses `127.0.0.1:11435` and `OLLAMA_MODELS=Models/ollama`. It starts on demand and remains available in the background, unloading idle model memory after five minutes. A custom loopback server can still be selected in Settings. The shared Ollama service on port 11434 is unaffected.

When recording or saving an image, sampled descriptions are written to `descriptions.jsonl` beside `predictions.jsonl`, with their original frame number, source time, model name, generation time, frame dimensions, exact evidence, `action_decision` and `reason`. A caption can finish after its recording is stopped, and still belongs to that recording's directory. Manual Stop can discard unfinished captions. Captions are displayed in the panel and saved as metadata; they are not burned into video frames.

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
- **Descriptions:** LLM or CIS 579 method, sampling interval, and the local vision model/server/timeout used in LLM mode.
- **Saving:** optional default save folder and **Record from start / save image result** for automatic export, including short files that might finish before you can click Record. Automatic video recording asks for a filename when opening the source. Cancelling starts preview without recording.

Change these settings before opening the source. A camera may choose a supported capture size or FPS different from the request. Model checkpoints download on first use; Nano, Small, Medium, and Large detection checkpoints are cached on this PC.

Images, video files, direct network streams, and Raspberry Pi CSI cameras are also available in the input selector. **Video** means an existing input video to play, and only that input mode needs an existing video file. Use **Webcam** for live camera viewing or recording. Video files are processed as fast as possible.

## Output and timing

Videos are saved at your selected MP4 filename. In **Both** mode, choosing `clip.mp4` saves annotations there and clean frames in `clip_original.mp4`. A companion `clip_details` folder contains `predictions.jsonl`, `settings.json`, and `run.json`. Image exports use timestamped folders with `.jpg` files. Videos are silent. Per-frame JSON records say whether inference was enabled and contain predictions when available. Network stream URLs are omitted from saved settings.

Live capture continuously reads the newest camera frame to avoid old frames accumulating. Recording uses the frames processed by the app. When inference or encoding is slower than capture, some captured frames are omitted and the previous processed frame is repeated in the recording to preserve elapsed playback time. Original-only recording with inference off uses the lighter camera-only path. Requested camera FPS is not a guaranteed processing rate. The displayed FPS is the measured processing rate over recent frames.

Saved videos use the capture resolution. Preview snapshots are limited to 1280×900. File-video frame skipping adjusts playback FPS accordingly. Stop normally finalizes MP4 files; a blocked worker is force-stopped after eight seconds, which can leave the current clip incomplete.

## Raspberry Pi / SBC installation

Copy `Run_CNN_LLM_Based_Descriptor.sh`, `launcher.py`, `main.py`, `setup_descriptor.py`, `requirements.txt`, and the `cnn_llm_based_descriptor/` and `assets/` folders into a `CNN_LLM_Based_Descriptor` folder on a **64-bit Linux desktop with Python 3.10+**, then open **Run_CNN_LLM_Based_Descriptor.sh**. Under **Installation & repair**, select **CPU · Raspberry Pi / SBC / PC** and click **Set up / repair**.

**Install Linux prerequisites…** installs Tk, venv, pip, and graphics libraries on apt-based Linux systems using the OS authentication dialog. If Tk is missing before the launcher can open, it offers a graphical bootstrap on systems with Zenity and PolicyKit; otherwise install Python Tk through the OS software manager. System package installation must finish before the window closes.

For a Raspberry Pi CSI camera, enable the CSI-support checkbox before installing Linux prerequisites. In the app, choose **Pi camera (CSI)** and index **0**. ARM environments expose system Picamera2/libcamera packages. Other boards can use compatible USB cameras. Pi/ARM hardware and CSI cameras have not been physically tested; the CPU path has been tested on this PC.

Use **Settings → Camera → Raspberry Pi / SBC preset** to select Nano, CPU, 320-pixel inference, two CPU threads, and frame stride 2. Larger models need more memory and may be too slow on small boards. No board-specific NPU acceleration is included.

Copy the `Models/` folder separately for offline use. Do not copy `.venv/`, `.environments/`, or `launcher_settings.json` between computers.

## Installation management

GUI setup offers existing-package reuse, CPU-only PyTorch, or NVIDIA CUDA packages. CUDA 13.0 matches this PC; CUDA 12.8 is also offered. Driver compatibility is described in [PyTorch's installation guide](https://pytorch.org/get-started/locally/). Setup does not change GPU drivers.

New environments are verified before activation; failed or cancelled installs leave the working environment available. The active environment is recorded in `launcher_settings.json`, with the original `.venv/` also supported. Repairs retain prior working environments and can require additional disk space.

## Verification

Both methods are checked for strict object grounding, unknown/contradictory scene rejection, and no-detection/classification behavior. The probability calculations are checked against direct arithmetic; UCS is checked against exhaustive plans for a small case and for its target bound. GUI checks cover both methods, icons, decision highlights, and minimum window sizes.

The descriptor’s 31 regression checks passed before the requested test-file cleanup. The refreshed launcher was checked at 800×740, 700×520 and 640×480 in ready, error and busy states. The main app was checked at 1440×900, 1100×700, 1024×600 and 760×520, including both decision highlights and scrolling settings. GitHub Actions runs syntax, launcher-layout and descriptor smoke checks without downloading models.

A real CUDA YOLO → quantized Qwen frame check verifies the local model pipeline. Raspberry Pi/ARM camera hardware remains untested.

Runtime libraries may recreate caches when the app is used. Git excludes local environments, model weights, settings, logs, and recordings.

## Project icon

The app and launcher use an original detection-frame, eye, and speech-bubble mark. Editable source: [`assets/icon.svg`](assets/icon.svg); PNG exports at 64 and 256 pixels keep Tk startup independent of imaging libraries.

## References

[YOLO26 models](https://docs.ultralytics.com/models/yolo26/) · [Prediction API](https://docs.ultralytics.com/modes/predict/) · [Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) · [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)

[Linux camera capabilities](https://cdn.kernel.org/doc/html/latest/userspace-api/media/v4l/vidioc-querycap.html) · [OpenCV camera backends](https://docs.opencv.org/4.13.0/d8/dfe/classcv_1_1VideoCapture.html)

Ultralytics code/models have their own AGPL-3.0 / enterprise terms. Dependencies retain their respective licenses.

## Automated checks

GitHub Actions contains the smoke checks directly in its workflow; a separate test folder is not shipped. Local verification can use `python3 -m compileall -q cnn_llm_based_descriptor main.py launcher.py setup_descriptor.py` and `bash -n Run_CNN_LLM_Based_Descriptor.sh`.

Repository: [CNN_LLM_Based_Descriptor](https://github.com/denizkarya1999/CNN_LLM_Based_Descriptor).
