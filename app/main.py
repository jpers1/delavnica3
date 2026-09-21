"""FastAPI service for the CPU-only Ultralytics YOLO web detection demo."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import anyio
import ultralytics
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from .detector import Detector, ImageDecodeError, decode_image
from .schemas import (
    BoxModel,
    DetectionModel,
    DetectResponse,
    ImageModel,
    InfoResponse,
)

logger = logging.getLogger("yolo_demo")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# --- Lightweight environment configuration -----------------------------
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolo26n.pt")
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_MAX_DETECTIONS = int(os.environ.get("YOLO_MAX_DETECTIONS", "100"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# Process-wide detector: the model is loaded exactly once at startup.
detector: Detector | None = None


def build_detector() -> Detector:
    return Detector(
        model_name=YOLO_MODEL,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        max_detections=YOLO_MAX_DETECTIONS,
    )


def create_app(detector_factory: Callable[[], Detector] | None = None) -> FastAPI:
    """Build the FastAPI app.

    ``detector_factory`` is a test seam: API contract tests inject a fake
    detector so they never download model weights or run real inference.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        global detector
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        started = time.perf_counter()
        detector = (detector_factory or build_detector)()
        logger.info("Inference device: CPU (forced; GPU never used)")
        logger.info("Model: %s", detector.model_name)
        logger.info("Backend: PyTorch CPU")
        logger.info("Ultralytics version: %s", ultralytics.__version__)
        logger.info("Inference image size: %s", detector.imgsz)
        logger.info("Confidence threshold: %s", detector.conf)
        logger.info("Model ready in %.0f ms", (time.perf_counter() - started) * 1000.0)
        logger.info(
            "Listening on http://%s:%s (bind with uvicorn, e.g. --host 0.0.0.0 --port 8000)",
            os.environ.get("HOST", "0.0.0.0"),
            os.environ.get("PORT", "8000"),
        )
        yield
        detector = None

    app = FastAPI(title="CPU-only YOLO web detection demo", lifespan=lifespan)

    # Serialize inference: one shared model, one CPU, no overlapping runs.
    inference_lock = asyncio.Lock()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/info", response_model=InfoResponse)
    async def info() -> InfoResponse:
        assert detector is not None  # set by lifespan before requests are served
        return InfoResponse(
            model=detector.model_name,
            backend=detector.backend,
            device=detector.device,
            imgsz=detector.imgsz,
            conf=detector.conf,
            max_detections=detector.max_detections,
            ultralytics_version=ultralytics.__version__,
        )

    @app.post("/api/detect", response_model=DetectResponse)
    async def detect(file: UploadFile = File(...)) -> DetectResponse:
        assert detector is not None
        # Read at most the cap + 1 byte so oversized uploads stay bounded.
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="image too large (10 MB max)")
        try:
            image = decode_image(data)
        except ImageDecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        started = time.perf_counter()
        try:
            async with inference_lock:
                detections, inference_ms = await anyio.to_thread.run_sync(
                    detector.infer, image
                )
        except Exception:
            logger.exception("inference failed")
            raise HTTPException(status_code=500, detail="inference failed") from None
        total_ms = (time.perf_counter() - started) * 1000.0

        logger.info(
            "detect: %dx%s image, %d detection(s), inference %.1f ms, total %.1f ms",
            image.width,
            image.height,
            len(detections),
            inference_ms,
            total_ms,
        )
        return DetectResponse(
            image=ImageModel(width=image.width, height=image.height),
            model=detector.model_name,
            device=detector.device,
            inference_ms=round(inference_ms, 1),
            detections=[
                DetectionModel(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=round(det.confidence, 4),
                    box=BoxModel(
                        x1=round(det.x1, 1),
                        y1=round(det.y1, 1),
                        x2=round(det.x2, 1),
                        y2=round(det.y2, 1),
                    ),
                )
                for det in detections
            ],
        )

    # Serve the demo page + assets last so API routes take priority.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
