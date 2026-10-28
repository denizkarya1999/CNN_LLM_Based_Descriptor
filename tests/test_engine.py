from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cnn_llm_based_descriptor.config import RunConfig
from cnn_llm_based_descriptor.engine import infer


class EngineIntegrationTests(unittest.TestCase):
    def run_source(self, source_kind="Image", describe_enabled=True, failure=False, cancel=False, inference_enabled=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        frame = np.zeros((80, 100, 3), np.uint8)
        source = root / ("sample.png" if source_kind == "Image" else "sample.mp4")
        if source_kind == "Image":
            cv2.imwrite(str(source), frame)
        else:
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (100, 80))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(frame)
            writer.release()
        weights = root/"weights.pt"
        weights.touch()
        records = [{"name": "person", "confidence": .9, "box": {"x1": 5, "y1": 10, "x2": 40, "y2": 70}}]
        result = SimpleNamespace(probs=None, names={0: "person"}, speed={"inference": 1},
                                 plot=lambda **kwargs: frame, summary=lambda: records)
        model = SimpleNamespace(task="detect", names={0: "person"}, predict=lambda *a, **kw: [result])
        cfg = replace(RunConfig(), source_kind=source_kind, source=str(source), device="cpu", half=False,
                      custom_model=str(weights), save_output=True, output_dir=str(root/"outputs"),
                      describe_enabled=describe_enabled, inference_enabled=inference_enabled)
        messages, previews = queue.Queue(), queue.Queue(2)
        stop = threading.Event()
        modules = {"torch": SimpleNamespace(set_num_threads=lambda count: None, inference_mode=nullcontext),
                   "ultralytics": SimpleNamespace(YOLO=lambda path: model),
                   "ultralytics.cfg": SimpleNamespace(DEFAULT_CFG_DICT={})}
        caption = {"text": "A person on the left, 90% detector confidence.", "evidence": records, "model": "fake"}
        def cancel_description(*args):
            stop.set()
            return caption
        with patch.dict("sys.modules", modules), patch(
            "cnn_llm_based_descriptor.descriptor.OllamaVisionClient.describe",
            side_effect=RuntimeError("Model unavailable") if failure else cancel_description if cancel else None,
            return_value=caption,
        ) as describe:
            infer(cfg, messages, previews, stop, threading.Event())
            self.assertEqual(describe.called, describe_enabled and inference_enabled)
        packets = []
        while not messages.empty():
            packets.append(messages.get_nowait())
        self.assertEqual(packets[-1]["type"], "done")
        self.assertEqual(packets[-1]["outcome"], "stopped" if cancel else "completed")
        directory = next((root/"outputs").iterdir())
        return packets, directory

    def test_still_waits_for_caption_and_records_evidence(self):
        packets, directory = self.run_source()
        caption = next(p for p in packets if p["type"] == "description")
        self.assertEqual(caption["frame"], 1)
        self.assertIsNone(caption["source_seconds"])
        self.assertEqual(json.loads((directory/"descriptions.jsonl").read_text())["frame"], 1)
        self.assertTrue((directory/"annotated.jpg").exists())

    def test_disabled_descriptions_preserve_detector_output(self):
        packets, directory = self.run_source(describe_enabled=False)
        self.assertFalse(any(p["type"] == "description" for p in packets))
        self.assertFalse((directory/"descriptions.jsonl").exists())
        self.assertEqual(json.loads((directory/"predictions.jsonl").read_text())["predictions"][0]["confidence"], .9)

    def test_model_failure_does_not_fail_detector_or_recording(self):
        packets, directory = self.run_source(failure=True)
        self.assertEqual(sum(p["type"] == "description_error" for p in packets), 1)
        self.assertEqual(json.loads((directory/"run.json").read_text())["status"], "completed")

    def test_video_eof_flushes_pending_caption(self):
        packets, directory = self.run_source("Video")
        self.assertEqual(packets[-1]["frames"], 4)
        caption = next(p for p in packets if p["type"] == "description")
        self.assertEqual(caption["source_seconds"], 0)
        self.assertTrue((directory/"annotated.mp4").exists())

    def test_stop_during_eof_wait_reports_stopped_and_discards_caption(self):
        packets, directory = self.run_source(cancel=True)
        self.assertFalse(any(p["type"] == "description" for p in packets))
        self.assertFalse((directory/"descriptions.jsonl").exists())
        self.assertEqual(json.loads((directory/"run.json").read_text())["status"], "stopped")

    def test_inference_off_preserves_raw_image_without_calling_model(self):
        packets, directory = self.run_source(inference_enabled=False)
        self.assertFalse(any(p["type"] == "description" for p in packets))
        self.assertTrue((directory/"original.jpg").exists())
        prediction = json.loads((directory/"predictions.jsonl").read_text())
        self.assertFalse(prediction["inference"])
        self.assertEqual(prediction["predictions"], [])


if __name__ == "__main__":
    unittest.main()
