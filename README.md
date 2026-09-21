# YOLO Web Detection Demo (CPU-only)

A browser-accessible object-detection demo built on **Ultralytics YOLO** that runs entirely on
**CPU** — no GPU, no CUDA, no training, no custom dataset. A plain HTML5 page captures webcam
frames (or an uploaded image) and sends them over HTTP to a FastAPI service; the service runs
inference with a pretrained **`yolo26n`** detection model and returns structured JSON
detections, which the browser draws back over the live video as bounding boxes with class
labels and confidence scores.

> **Status: specification complete — implementation in progress.**
> The project spec, architecture, API contract, and work plan are finalized in
> [`AGENTS.md`](AGENTS.md). The service, web client, and tests are being implemented against
> that spec now, targeting a live demo on **2026-09-22**.

---

## Project status

| Area | Status |
|---|---|
| Project specification & constraints ([`AGENTS.md`](AGENTS.md)) | ✅ Done |
| API design & response contract | ✅ Done (defined in spec) |
| Documentation (this README) | ✅ Done |
| FastAPI service + Ultralytics YOLO detector | ⏳ In progress (next work item) |
| HTML5 webcam client with bounding-box overlay | ⏳ Planned |
| Automated API tests | ⏳ Planned |
| Performance benchmarking (measured latency/FPS) | ⏳ Planned |
| Optional OpenVINO CPU optimization | 🧪 Optional, after the baseline path works |

Status labels follow the project convention: ✅ implemented · ⏳ planned/in progress ·
🧪 optional/experimental · ❌ out of scope.

---

## Why this project

Modern object detection is usually shown on GPUs. This demo shows the other side: an ordinary
Linux/WSL2 machine with **no GPU at all** can still serve useful real-time object detection to
a browser. The value is in proving the full chain works end-to-end on commodity hardware:

```
Browser webcam
    |
    | JPEG image over HTTP
    v
FastAPI service
    |
    v
Ultralytics YOLO26n
    |
    v
CPU inference
    |
    v
JSON detections
    |
    v
Browser bounding-box overlay
```

### Design principles

1. **CPU-only, enforced.** No NVIDIA hardware, CUDA, cuDNN, TensorRT, or GPU PyTorch wheels.
   The service reports its effective inference device so CPU execution is provable, not assumed.
2. **No training, no custom data.** Official pretrained `yolo26n` weights (Ultralytics 8.4.157).
3. **Boring reliability over features.** One model instance for the whole process, serialized
   inference, bounded uploads, no unbounded request queues.
4. **Measured, not claimed.** Latency and FPS figures are reported only when actually measured
   on the demo machine.

---

## Planned quickstart

> The commands below are the target setup and will be verified end-to-end (and marked as such
> in this README) once the implementation work item lands.

### Native Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### WSL2

Same commands as above. The primary demo path is
`Windows browser → http://localhost:8000 → WSL2 FastAPI`.

### Run the demo

1. Start the server as above.
2. Open `http://localhost:8000` in the browser.
3. Allow the camera permission prompt.
4. Press **Start Detection**.
5. Stand in view and watch `person` (and other classes) get boxed and labeled live.
6. Watch the measured request/inference latency update in the UI.

### Configuration

All settings are environment-variable friendly:

```text
YOLO_MODEL=yolo26n.pt
YOLO_BACKEND=auto        # auto = exported OpenVINO model if present, else PyTorch CPU
YOLO_IMGSZ=640
YOLO_CONF=0.25
YOLO_MAX_DETECTIONS=100
HOST=0.0.0.0
PORT=8000
```

On startup the service logs the Ultralytics version, model, backend, device (`CPU`),
inference image size, and confidence threshold.

---

## API design

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves the HTML5 webcam demo page |
| `GET /health` | Service readiness (`{"status": "ok"}`) |
| `GET /api/info` | Model, backend, device, inference size |
| `POST /api/detect` | Accepts one image (`multipart/form-data`, JPEG/PNG), returns JSON detections |

Example `/api/detect` response:

```json
{
  "image": { "width": 640, "height": 480 },
  "model": "yolo26n",
  "device": "cpu",
  "inference_ms": 84.3,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "box": { "x1": 120.4, "y1": 54.8, "x2": 410.2, "y2": 470.1 }
    }
  ]
}
```

Boxes use the uploaded image's coordinate system, with validated
`0 <= x1 < x2 <= width` and `0 <= y1 < y2 <= height`; confidences are in `[0, 1]`.

## Browser client

Plain HTML5 + CSS + vanilla JavaScript — no frontend framework. The page uses
`navigator.mediaDevices.getUserMedia()` for the camera, an offscreen `<canvas>` for JPEG frame
capture, and `fetch()` to post frames to `/api/detect`. Frames are sent with **backpressure**:
capture → send → await response → draw → schedule next frame — so an unbounded request queue
can never build up, and no frame is sent while the previous inference is outstanding.

**Webcam security note:** browsers only grant camera access in a secure context.
`http://localhost` works for local development; for another device on the LAN, plain HTTP may
be blocked by the browser. The supported demo paths are therefore: (1) browser and service on
the same machine via `localhost`, (2) HTTPS termination for remote clients, or (3) the static
image-upload fallback built into the page.

## Tests & evidence

Automated tests (no webcam required, local fixture image):

- `GET /health` returns 200 with the expected status
- `GET /api/info` reports `device: cpu`
- `POST /api/detect` cleanly rejects invalid image data
- `POST /api/detect` on a known fixture returns valid JSON with in-range boxes and confidences

A benchmark script reports model/backend, input resolution, logical-core count, warm-up
iterations, mean/median inference latency, and approximate FPS — after warm-up, never from
marketing numbers.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| A — Make it correct | Service starts, model loads, one known image detects, CPU confirmed | ⏳ In progress |
| B — Make it interactive | Webcam client, overlay, timing display, upload fallback | ⏳ Planned |
| C — Optimize | OpenVINO export/benchmark; keep the faster reliable CPU backend as default | 🧪 Optional |

Explicit non-goals for the first release: model training/fine-tuning, object tracking, video
recording, authentication, databases, Docker requirement, and any other framework beyond the
stack above.

---

## References

- Ultralytics models: https://docs.ultralytics.com/models/
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- OpenVINO export: https://docs.ultralytics.com/integrations/openvino/
- Ultralytics on PyPI: https://pypi.org/project/ultralytics/

## License

See [`LICENSE`](LICENSE).
