# YOLO Web Detection Demo (CPU-only)

A browser-accessible object-detection demo built on **Ultralytics YOLO** that runs
entirely on **CPU** — no GPU, no CUDA, no training, no custom dataset. A plain HTML5
page captures webcam frames (or an uploaded image) and sends them over HTTP to a
FastAPI service; the service runs inference with the pretrained **`yolo26n`**
detection model and returns structured JSON detections, which the browser draws back
over the live video as bounding boxes with class labels and confidence scores.

```
Browser webcam / uploaded image
    |
    | JPEG over HTTP (multipart)
    v
FastAPI (uvicorn)
    |
    v
Ultralytics YOLO26n  (yolo26n.pt, pretrained)
    |
    v
CPU inference (device forced to cpu)
    |
    v
JSON detections
    |
    v
Browser bounding-box overlay
```

Status labels follow the project convention: ✅ implemented · ⏳ planned/in progress ·
🧪 optional/experimental · ❌ out of scope.

| Area | Status |
|---|---|
| Project specification & constraints ([`AGENTS.md`](AGENTS.md)) | ✅ Done |
| FastAPI service + Ultralytics YOLO detector (CPU-forced) | ✅ Implemented |
| HTML5 webcam client with bounding-box overlay + timing | ✅ Implemented |
| Static image upload fallback | ✅ Implemented |
| Automated API tests (health / info / detect contract) | ✅ Implemented, passing |
| Performance baseline (measured, see below) | ✅ Measured |
| Optional OpenVINO CPU optimization | ⏳ Planned (next work order, not implemented) |
| Object tracking, training, auth, database, Docker | ❌ Out of scope |

---

## Quickstart (Linux / WSL2)

Tested on **Ubuntu 26.04 under WSL2** with Python 3.14 and
`ultralytics==8.4.157`.

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU-only PyTorch + torchvision (must come from the CPU index; the default
# PyPI wheels bundle CUDA libraries. On a GPU-less machine they "work" but
# bloat the install; the CPU builds keep this project honestly CPU-only).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

First start downloads the official `yolo26n.pt` weights (~5 MB) from
Ultralytics' asset releases; afterwards they are cached locally.

## Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Startup log (CPU operation is stated explicitly):

```text
Inference device: CPU (forced; GPU never used)
Model: yolo26n.pt
Backend: PyTorch CPU
Ultralytics version: 8.4.157
Inference image size: 640
Confidence threshold: 0.25
```

You can also confirm the effective device at any time via
[`GET /api/info`](#api).

## Run the demo

1. Start the server as above.
2. Open **http://localhost:8000** in a browser on the same machine.
3. Allow the camera permission prompt.
4. Press **Start Detection**.
5. Stand in view — `person` (and other COCO classes) get boxed with
   label + confidence.
6. Watch the measured `request ms` / `server ms` / `~N img/s` line update.
7. **Stop Detection** cancels the loop cleanly.

If the camera is unavailable, use the **static image fallback** panel: pick a
JPEG/PNG and press **Detect image**. It hits the same `/api/detect` endpoint.

### WSL2

Run the Linux commands above inside WSL2. The primary demo path is
**Windows browser → `http://localhost:8000` → WSL2 FastAPI**; WSL2 forwards
Windows' `localhost` to the distro by default. The service was verified running
inside WSL2 and reachable on `localhost:8000` from the distro side; the
Windows-side browser leg has not been tested in this environment (no Windows
host browser available here). If localhost forwarding is disabled on your host,
the minimal fallback is to use the Windows-visible WSL IP (from `ip addr`
inside WSL2) as the page origin — but then see the camera caveat below.

### Browser camera caveat (read before the demo)

Browsers only grant camera access in a **secure context**.
`http://localhost` is treated as secure by modern browsers, so the
same-machine flow works. From **another device on the LAN over plain HTTP**,
webcam access is generally **blocked** — that path is *not* supported/tested.
Supported options: (1) browser and service on the same machine via
`localhost`; (2) HTTPS termination for remote clients; (3) the static
image-upload fallback built into the page.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves the HTML5 webcam demo page |
| `GET /health` | Service readiness: `{"status": "ok"}` |
| `GET /api/info` | Model, backend, device, inference size, conf, ultralytics version |
| `POST /api/detect` | One image (`multipart/form-data`, field `file`, JPEG or PNG, ≤ 10 MB) → JSON detections |

`/api/info` response (CPU is reported and enforced):

```json
{
  "model": "yolo26n.pt",
  "backend": "pytorch",
  "device": "cpu",
  "imgsz": 640,
  "conf": 0.25,
  "max_detections": 100,
  "ultralytics_version": "8.4.157"
}
```

`/api/detect` response contract:

```json
{
  "image": { "width": 1280, "height": 720 },
  "model": "yolo26n.pt",
  "device": "cpu",
  "inference_ms": 105.2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.9005,
      "box": { "x1": 742.4, "y1": 41.0, "x2": 1151.3, "y2": 709.0 }
    }
  ]
}
```

Boxes use the uploaded image's coordinate system with
`0 <= x1 < x2 <= width` and `0 <= y1 < y2 <= height`; confidences are in
`[0, 1]`. An empty `detections` array is a normal success, not an error.
Errors are clean JSON (`400` invalid/unsupported image, `413` too large,
`500` inference failure) — no Python tracebacks are exposed.

## Configuration

Environment variables (simple defaults, no `.env` file needed):

```text
YOLO_MODEL=yolo26n.pt
YOLO_IMGSZ=640
YOLO_CONF=0.25
YOLO_MAX_DETECTIONS=100
MAX_UPLOAD_BYTES=10485760
HOST=0.0.0.0        # used in the startup log only; bind with uvicorn
PORT=8000
```

The inference device is **always CPU** — it is hard-wired in the detector, not
selected at runtime, so a GPU appearing on the host is never used.

## Browser client

Plain HTML5 + CSS + vanilla JavaScript (no build step). `navigator.mediaDevices
getUserMedia()` for the camera, an offscreen `<canvas>` for JPEG frame capture,
`fetch()` for the HTTP round-trip, and an overlay `<canvas>` for boxes.

**Backpressure:** the loop is `capture → send → await response → draw → wait →
repeat`; a new frame is never sent while a previous request is outstanding, so
no unbounded queue can build up against the CPU-only backend. A client-side
15 s request timeout and an error-pause (1 s) guard against hammering a broken
backend. Start/Stop buttons are state-managed (Start is disabled while running).

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Coverage (11 tests, all passing at time of writing):

- `GET /health` returns 200 with the expected status
- `GET /api/info` reports `device: cpu` and the expected runtime fields
- `POST /api/detect` rejects: invalid bytes, empty upload, GIF (unsupported
  format), oversized upload (10 MB cap), missing file field
- `POST /api/detect` on a JPEG/PNG returns the full contract: image size,
  `device: cpu`, numeric in-bounds boxes, confidences in `[0, 1]`, and an
  empty-detection success case

API tests inject a fake detector at the documented test seam
(`create_app(detector_factory=...)`) so they stay fast and never need model
weights. The real model path is verified separately — see below.

## Performance baseline (measured, not claimed)

Measured with `scripts/benchmark.py` (3 warm-up + 10 timed inferences,
`imgsz=640`, `conf=0.25`, 8 logical cores, `bus.jpg` 810×1080 sample image,
Ubuntu 26.04 / WSL2, PyTorch CPU 2.14.0+cpu):

```text
model: yolo26n.pt · backend: pytorch · device: cpu
mean latency:   80.4 ms
median latency: 71.2 ms
approx model-only FPS: 14.1
```

End-to-end HTTP (localhost, same image, warmed server): ~70–180 ms per
request. This comfortably supports the 2–5 requests/s demo target on this
machine; **re-measure on the actual demo machine** — numbers are hardware-
specific. First request after server start is slower (model warm-up).

## Known limitations

- **Browser/webcam path not verified in this environment** (headless, no
  browser, no camera). The HTTP contract the page consumes is fully
  verified; the JavaScript rendering path needs a real browser check before
  the live demo.
- WSL2 → Windows-browser `localhost` leg not tested here (no Windows host
  browser); WSL2's default localhost forwarding is expected to work.
- LAN-over-plain-HTTP webcam access is **not** supported (browser secure
  context); use localhost, HTTPS, or the image-upload fallback.
- Single process, one model instance, serialized inference — intended for a
  demo, not a high-throughput service. No multi-worker deployment.
- `yolo26s` / larger variants are not evaluated (out of scope for this PR).

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| A — Make it correct | Service starts, model loads, known image detects, CPU confirmed | ✅ Done |
| B — Make it interactive | Webcam client, overlay, timing display, upload fallback | ✅ Done |
| C — Optimize | OpenVINO export/benchmark via Ultralytics; keep the faster reliable CPU backend | ⏳ Next work order |

Explicit non-goals for this release: model training/fine-tuning, object
tracking, video recording, authentication, databases, Docker requirement,
WebSocket/WebRTC, and any GPU path.

## References

- Ultralytics models: https://docs.ultralytics.com/models/
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- OpenVINO export: https://docs.ultralytics.com/integrations/openvino/
- Ultralytics on PyPI: https://pypi.org/project/ultralytics/

## License

See [`LICENSE`](LICENSE).
