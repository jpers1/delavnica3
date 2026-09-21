"""CPU-only Ultralytics YOLO inference wrapper for the demo service."""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Only the formats the demo contract promises (browser JPEG frames, JPEG/PNG uploads).
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an allowed image."""


@dataclass(frozen=True)
class Detection:
    """One detection in uploaded-image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def decode_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an RGB PIL image.

    Raises ImageDecodeError for undecodable data or unsupported formats.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force a full decode so truncated/corrupt files fail here
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodeError("uploaded data is not a readable image") from exc

    if image.format not in ALLOWED_IMAGE_FORMATS:
        raise ImageDecodeError(
            f"unsupported image format {image.format!r}; upload a JPEG or PNG"
        )

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class Detector:
    """Loads a pretrained Ultralytics YOLO model once and runs CPU-only inference."""

    def __init__(
        self,
        model_name: str,
        imgsz: int = 640,
        conf: float = 0.25,
        max_detections: int = 100,
    ) -> None:
        self.model_name = model_name
        self.imgsz = imgsz
        self.conf = conf
        self.max_detections = max_detections
        self.backend = "pytorch"
        # This project is CPU-only by design: the device is hard-wired to CPU
        # even if a GPU happens to exist on the host.
        self.device = "cpu"
        self.model = YOLO(model_name)
        logger.info("Loaded model %s for CPU inference", model_name)

    def infer(self, image: Image.Image) -> tuple[list[Detection], float]:
        """Run one CPU inference on a PIL image.

        Returns ``(detections, inference_ms)``. Boxes are clamped to the
        uploaded image's bounds; degenerate boxes are dropped.
        """
        width, height = image.size
        started = time.perf_counter()
        results = self.model.predict(
            np.asarray(image),
            imgsz=self.imgsz,
            conf=self.conf,
            device="cpu",
            verbose=False,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for cls, conf, (x1, y1, x2, y2) in zip(boxes.cls, boxes.conf, boxes.xyxy):
                if len(detections) >= self.max_detections:
                    break
                class_id = int(cls)
                detection = Detection(
                    class_id=class_id,
                    class_name=result.names[class_id],
                    confidence=float(conf),
                    x1=max(0.0, float(x1)),
                    y1=max(0.0, float(y1)),
                    x2=min(float(width), float(x2)),
                    y2=min(float(height), float(y2)),
                )
                if detection.x1 < detection.x2 and detection.y1 < detection.y2:
                    detections.append(detection)
        return detections, inference_ms
