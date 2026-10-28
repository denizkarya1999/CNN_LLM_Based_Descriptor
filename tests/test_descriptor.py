import base64
from dataclasses import replace
import io
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

from yolo_llm_visual_descriptor.config import RunConfig
from yolo_llm_visual_descriptor.descriptor import (
    FrameDescriber, OllamaVisionClient, build_prompt, detection_evidence,
    request_json, validate_endpoint,
)

FRAME = np.full((80, 100, 3), (0, 0, 255), dtype=np.uint8)
RECORDS = [{"name": "person", "confidence": .91, "box": {"x1": 0, "y1": 0, "x2": 30, "y2": 20}}]


class EvidenceTests(unittest.TestCase):
    def test_boxes_scores_sorting_and_truncation(self):
        records = RECORDS + [{"name": "car", "confidence": .99,
                              "box": {"x1": 70, "y1": 60, "x2": 120, "y2": 90}}]
        evidence, total = detection_evidence(records, 100, 80, 1)
        self.assertEqual(total, 2)
        self.assertEqual(evidence[0]["confidence_percent"], 99)
        self.assertEqual(evidence[0]["box_xyxy_normalized"], [.7, .75, 1, 1])
        self.assertEqual(evidence[0]["position"], "bottom-right")
        self.assertIn('"omitted_predictions": 1', build_prompt(evidence, total, 100, 80, .25, "Object detection"))

    def test_oriented_boxes_classification_and_invalid_data(self):
        records = [
            {"name": "plane", "confidence": .8, "box": {"x1": 40, "y1": 10, "x2": 80, "y2": 30, "x3": 60, "y3": 60, "x4": 20, "y4": 40}},
            {"name": "scene", "confidence": .7},
            {"confidence": float("nan")}, {"confidence": 1.5},
            {"confidence": .5, "box": {"x1": 1, "y1": 1, "x2": 1, "y2": 2}},
        ]
        evidence, total = detection_evidence(records, 100, 80, 20)
        self.assertEqual(total, 2)
        self.assertEqual(evidence[0]["box_xyxy_pixels"], [20, 10, 80, 60])
        self.assertNotIn("box_xyxy_pixels", evidence[1])

    def test_endpoint_and_settings(self):
        for valid in ("http://127.0.0.1:11434", "http://localhost:11434/", "http://[::1]:11434"):
            validate_endpoint(valid)
        for invalid in ("https://example.com", "http://localhost@evil.com", "http://localhost:abc",
                        "http://127.0.0.1/api", "file:///tmp/x", "http://localhost?x=y"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_endpoint(invalid)
        for field, value in (("describe_interval", float("nan")), ("describe_timeout", 0),
                             ("describe_max_objects", 100), ("describe_model", "qwen3-vl:2b-cloud")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(RunConfig(), **{field: value}).validate_descriptor()
        replace(RunConfig(), describe_enabled=False, describe_url="bad").validate_descriptor()


class ClientTests(unittest.TestCase):
    @patch("yolo_llm_visual_descriptor.descriptor.request_json")
    def test_sends_real_frame_and_exact_evidence(self, request):
        request.side_effect = [
            {"capabilities": ["vision"], "details": {"quantization_level": "Q4_K_M"}},
            {"message": {"content": "A person is at the top-left (91% detector confidence)."}},
        ]
        result = OllamaVisionClient(RunConfig()).describe(FRAME, RECORDS)
        payload = request.call_args.args[2]
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(payload["messages"][0]["images"][0]), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, FRAME.shape)
        self.assertGreater(decoded[0, 0, 2], 240)  # BGR/RGB color is preserved.
        self.assertLess(decoded[0, 0, 0], 10)
        self.assertIn('"confidence_percent": 91.0', payload["messages"][0]["content"])
        self.assertFalse(payload["stream"])
        self.assertNotIn("think", payload)  # Non-thinking vision models reject unsupported controls.
        self.assertEqual(result["evidence"][0]["position"], "top-left")

    @patch("yolo_llm_visual_descriptor.descriptor.request_json")
    def test_no_detections_and_empty_response(self, request):
        request.side_effect = [
            {"capabilities": ["vision"], "details": {"quantization_level": "Q4_K_M"}},
            {"message": {"content": "  "}},
        ]
        with self.assertRaisesRegex(ValueError, "empty"):
            OllamaVisionClient(RunConfig()).describe(FRAME, [])
        self.assertIn('"predictions": []', request.call_args.args[2]["messages"][0]["content"])

    @patch("yolo_llm_visual_descriptor.descriptor.request_json")
    def test_rejects_text_only_unquantized_and_remote_models(self, request):
        for metadata in ({"capabilities": ["completion"]},
                         {"capabilities": ["vision"], "details": {"quantization_level": "F16"}},
                         {"capabilities": ["vision"], "remote_host": "https://ollama.com"}):
            request.return_value = metadata
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                OllamaVisionClient(RunConfig()).describe(FRAME, RECORDS)

    @patch("yolo_llm_visual_descriptor.descriptor.build_opener")
    def test_missing_model_and_connection_errors(self, opener):
        for error, message in ((HTTPError("http://localhost", 404, "missing", {}, io.BytesIO()), "missing"),
                               (URLError("offline"), "reach Ollama"), (TimeoutError(), "timed out")):
            opener.return_value.open.side_effect = error
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, message):
                request_json("http://localhost:11434", "/api/chat", {}, 5)


class BlockingClient:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = 0

    def describe(self, frame, records):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(3):
            raise TimeoutError("Test did not release request")
        return {"text": "A person (91% detector confidence).", "evidence": records, "model": "fake"}


class WorkerTests(unittest.TestCase):
    def make_worker(self, client=None):
        messages, stop, enabled = queue.Queue(), threading.Event(), threading.Event()
        enabled.set()
        worker = FrameDescriber(RunConfig(), messages, stop, enabled, client or BlockingClient())
        self.addCleanup(worker.close)
        return worker, messages, stop, enabled

    def submit(self, worker, index, directory=None):
        worker.next_at = 0
        return worker.submit(FRAME, RECORDS, index, index/30, 1.0, directory)

    def test_bounded_latest_queue_and_save_frame_identity(self):
        worker, messages, _, _ = self.make_worker()
        with tempfile.TemporaryDirectory() as directory:
            self.submit(worker, 1, directory)
            self.assertTrue(worker.client.entered.wait(1))
            for i in range(2, 8):
                self.submit(worker, i, directory)
            self.assertEqual(worker.pending.qsize(), 1)
            worker.client.release.set()
            worker.close(drain=True)
            packets = [messages.get_nowait(), messages.get_nowait()]
            self.assertEqual([p["frame"] for p in packets], [1, 7])
            saved = [json.loads(line) for line in (Path(directory)/"descriptions.jsonl").read_text().splitlines()]
            self.assertEqual([p["frame"] for p in saved], [1, 7])
            self.assertEqual(saved[1]["source_seconds"], 7/30)

    def test_invalidation_discards_inflight_caption(self):
        worker, messages, _, enabled = self.make_worker()
        self.submit(worker, 1)
        self.assertTrue(worker.client.entered.wait(1))
        enabled.clear()
        worker.invalidate()
        enabled.set()
        worker.client.release.set()
        worker.close(drain=True)
        self.assertTrue(messages.empty())

    def test_stop_does_not_wait_for_inflight_request(self):
        worker, messages, stop, _ = self.make_worker()
        self.submit(worker, 1)
        self.assertTrue(worker.client.entered.wait(1))
        stop.set()
        started = time.monotonic()
        worker.close(drain=True)
        self.assertLess(time.monotonic()-started, .7)
        worker.client.release.set()
        worker.thread.join(1)
        self.assertTrue(messages.empty())

    def test_failure_is_reported_once_and_does_not_raise_in_submit(self):
        class FailedClient:
            def describe(self, frame, records):
                raise TimeoutError("model timeout")
        worker, messages, _, _ = self.make_worker(FailedClient())
        self.submit(worker, 1)
        self.assertEqual(messages.get(timeout=1)["type"], "description_error")
        self.assertFalse(self.submit(worker, 2))


if __name__ == "__main__":
    unittest.main()
