"""Measure CPU inference latency for the demo detector.

Loads the real model, warms it up, then times N inferences on a local image
and reports mean/median latency and approximate model-only FPS.

Usage:
    python scripts/benchmark.py [--model yolo26n.pt] [--imgsz 640] [--conf 0.25]
                                [--image PATH] [--warmup 3] [--iterations 10]

Without --image, uses a sample asset shipped inside the ultralytics package
(e.g. assets/bus.jpg). Never downloads random files.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.detector import Detector, decode_image  # noqa: E402


def find_default_image() -> Path:
    candidates = [
        Path(os.environ.get("BENCH_IMAGE", "")),
        Path("tests/fixtures/sample.jpg"),
    ]
    import ultralytics

    asset_dir = Path(ultralytics.__file__).parent / "assets"
    candidates += [asset_dir / name for name in ("bus.jpg", "zidane.jpg")]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise SystemExit(
        "No benchmark image found. Pass --image PATH or set BENCH_IMAGE."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("YOLO_MODEL", "yolo26n.pt"))
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("YOLO_IMGSZ", "640")))
    parser.add_argument("--conf", type=float, default=float(os.environ.get("YOLO_CONF", "0.25")))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    image_path = args.image or find_default_image()
    image = decode_image(image_path.read_bytes())

    detector = Detector(args.model, imgsz=args.imgsz, conf=args.conf)
    print("model:            ", detector.model_name)
    print("backend:          ", detector.backend)
    print("device:           ", detector.device)
    print("imgsz:            ", detector.imgsz)
    print("conf:             ", detector.conf)
    print("cpu logical cores:", os.cpu_count())
    print("image:            ", image_path, f"({image.width}x{image.height})")

    for _ in range(args.warmup):
        detector.infer(image)
    print(f"warm-up:          {args.warmup} iteration(s) (not timed)")

    latencies: list[float] = []
    last_count = 0
    for _ in range(args.iterations):
        detections, inference_ms = detector.infer(image)
        latencies.append(inference_ms)
        last_count = len(detections)

    median_ms = statistics.median(latencies)
    mean_ms = statistics.mean(latencies)
    print(f"iterations:       {args.iterations}")
    print(f"detected:         {last_count} object(s) on the sample image")
    print(f"mean latency ms:  {mean_ms:.1f}")
    print(f"median latency ms:{median_ms:.1f}")
    print(f"approx FPS:       {1000.0 / median_ms:.2f}")
    print(
        "Note: model-only figure; end-to-end HTTP latency is higher\n"
        "(upload transfer + decode + serialization)."
    )


if __name__ == "__main__":
    main()
