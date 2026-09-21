"""Shared fixtures for API contract tests.

The real YOLO model is exercised separately (manual smoke test +
scripts/benchmark.py). API tests inject a fake detector so they stay fast
and never require model weights or network access. The fake mirrors the
real detector's public interface and reports device="cpu", which is the
device contract these tests verify at the API boundary.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detector import Detection  # noqa: E402
from app.main import create_app  # noqa: E402


class FakeDetector:
    """Stands in for app.detector.Detector in API tests."""

    backend = "pytorch"
    device = "cpu"  # the CPU contract under test

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        max_detections: int = 100,
    ) -> None:
        self.model_name = model_name
        self.imgsz = imgsz
        self.conf = conf
        self.max_detections = max_detections

    def infer(self, image: Image.Image) -> tuple[list[Detection], float]:
        # Small probe images report "no objects"; larger images get one
        # fixed, in-bounds person box so contract tests have data to check.
        if image.width < 100 or image.height < 100:
            return [], 1.0
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.87,
                x1=10.0,
                y1=20.0,
                x2=float(image.width - 10),
                y2=float(image.height - 10),
            )
        ], 1.0


def _encode(image_format: str, width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(120, 90, 60))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture()
def client():
    app = create_app(detector_factory=lambda: FakeDetector())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_jpeg() -> bytes:
    return _encode("JPEG", 320, 240)


@pytest.fixture()
def sample_png() -> bytes:
    return _encode("PNG", 320, 240)


@pytest.fixture()
def sample_gif() -> bytes:
    return _encode("GIF", 320, 240)


@pytest.fixture()
def small_jpeg() -> bytes:
    return _encode("JPEG", 64, 48)
