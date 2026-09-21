# AGENTS.md

## Project: CPU-Only Ultralytics YOLO Web Detection Demo

### Status
This repository is a fast-turnaround demonstration project.

Target demo date: 2026-09-22.

The implementation must favor:
1. reliability,
2. simple setup,
3. CPU-only execution,
4. clear browser-visible results,
5. reproducible evidence,
6. minimal scope.

Do not expand the project into a general computer-vision platform.

---

## 1. Mission

Build a browser-accessible object-detection service using **Ultralytics YOLO** on a **CPU-only Linux or WSL2 environment**.

The application must:

- use an official pretrained Ultralytics detection model;
- require no training and no custom dataset;
- perform inference on CPU only;
- accept images from a browser over HTTP;
- support live webcam capture from an HTML5 page;
- return structured object detections;
- draw bounding boxes and class labels in the browser;
- be easy to start for a live demonstration;
- provide basic performance measurements.

The primary demonstration path is:

```text
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

---

## 2. Human Intent and Product Goal

The human wants a practical demonstration of modern object detection without any GPU.

This is not a training project.

This is not a model-development project.

This is not a CUDA project.

The value of the demo is to show that:

- an ordinary browser can act as the camera client;
- frames can be sent to a local/remote Linux or WSL2 service;
- Ultralytics YOLO can perform useful object detection using CPU resources;
- the server can expose a simple API similar in spirit to an inference endpoint;
- several detections per second can be demonstrated on reasonable multicore hardware.

A stable 2-5 browser inference requests per second is a useful demo target, but actual achieved throughput must be measured rather than claimed.

---

## 3. Current Technical Baseline

Baseline verified on 2026-09-21:

- Latest released Ultralytics model family: **YOLO26**
- Default model for this project: **`yolo26n.pt`**
- Ultralytics package baseline: **`ultralytics==8.4.157`**
- Primary runtime: Linux or Ubuntu under WSL2
- Python: prefer Python 3.11 or 3.12 unless the host already has another Ultralytics-supported Python version
- API server: FastAPI
- ASGI server: Uvicorn
- Frontend: plain HTML5 + CSS + vanilla JavaScript
- Camera API: `navigator.mediaDevices.getUserMedia()`
- Default image transport: JPEG image requests over HTTP
- Default inference size: `imgsz=640`
- Default confidence threshold: approximately `0.25`, configurable
- Default model task: object detection

Official upstream references:

- https://docs.ultralytics.com/models/
- https://docs.ultralytics.com/models/yolo26/
- https://docs.ultralytics.com/integrations/openvino/
- https://docs.ultralytics.com/modes/export/
- https://pypi.org/project/ultralytics/

If implementation begins substantially later than the baseline date, verify upstream compatibility before changing pinned versions.

Do not silently replace YOLO26 with a different model family.

---

## 4. Non-Negotiable CPU-Only Rule

The finished demo must work on a machine with **no usable GPU**.

The implementation must not require:

- NVIDIA hardware;
- CUDA;
- cuDNN;
- TensorRT;
- ROCm;
- DirectML;
- GPU-specific PyTorch builds;
- GPU-specific OpenVINO execution;
- cloud GPU inference.

Do not make CUDA installation part of setup.

Do not require `nvidia-smi`.

Do not make GPU availability a prerequisite.

If PyTorch is present, inference must explicitly use CPU where appropriate.

The application must expose or log the effective inference backend/device so the demo operator can prove CPU execution.

A startup log should clearly state something equivalent to:

```text
Inference device: CPU
Model: yolo26n
Backend: PyTorch CPU
```

or:

```text
Inference device: CPU
Model: yolo26n
Backend: OpenVINO CPU
```

Never report CPU-only success unless the service actually runs without GPU access.

---

## 5. Ultralytics Requirement

Ultralytics must remain the primary model API.

The normal Python interface should use:

```python
from ultralytics import YOLO
```

The initial correctness path should load:

```python
model = YOLO("yolo26n.pt")
```

For optimized CPU inference, an Ultralytics-exported OpenVINO model is allowed and encouraged after the basic path works:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="openvino")
```

The exported model must still be loaded through the Ultralytics API where practical:

```python
model = YOLO("yolo26n_openvino_model/")
```

OpenVINO is an execution optimization, not a replacement for Ultralytics.

Do not rewrite YOLO inference manually.

Do not replace Ultralytics with raw ONNX Runtime, raw OpenVINO APIs, OpenCV DNN, TensorFlow, Detectron2, or another framework unless the human explicitly changes the requirement.

---

## 6. Model Selection

Start with:

```text
yolo26n.pt
```

Reason:

- lowest-risk CPU choice;
- smallest standard detection variant;
- best chance of usable interactive throughput;
- adequate for a demo.

Do not start with `yolo26m`, `yolo26l`, or `yolo26x`.

`yolo26s.pt` may be evaluated only after `yolo26n.pt` works end-to-end.

If a larger model is tested, benchmark it against the nano model and report the measured latency/FPS tradeoff.

No model training is allowed in the first release.

No fine-tuning is allowed in the first release.

No custom dataset is required.

Use official pretrained weights.

---

## 7. Optimization Strategy

Implementation order matters.

### Phase A - Make it correct

First implement a working CPU path using the ordinary Ultralytics model.

Required proof:

- service starts;
- model loads;
- one known image can be submitted;
- detections are returned;
- class names are correct;
- coordinates are valid;
- CPU operation is confirmed.

### Phase B - Make it interactive

Add:

- webcam browser client;
- JPEG frame capture;
- adjustable frame-send interval;
- browser overlay;
- request latency display;
- server inference-time display.

### Phase C - Optimize

Only after Phase A and B work:

1. export `yolo26n.pt` using Ultralytics OpenVINO export;
2. run the exported model on CPU;
3. benchmark it;
4. keep the faster reliable backend as the demo default.

Do not perform speculative optimization before the end-to-end path works.

---

## 8. Proposed Repository Structure

Prefer a small repository:

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── detector.py
│   ├── schemas.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── styles.css
├── scripts/
│   ├── export_openvino.py
│   └── benchmark.py
└── tests/
    ├── test_health.py
    ├── test_detect.py
    └── test_detector.py
```

Keep the structure simpler if fewer files are sufficient.

Do not add React, Vue, Angular, Node.js, webpack, Vite, or another frontend toolchain for the first demo.

---

## 9. Server Architecture

Use FastAPI.

The service should load the model once during application startup or through a single process-wide detector object.

Do not reload the model for each request.

Minimum endpoints:

```text
GET  /
GET  /health
GET  /api/info
POST /api/detect
```

Recommended behavior:

### `GET /`

Return the HTML5 webcam demonstration page.

### `GET /health`

Return simple service readiness information.

Example:

```json
{
  "status": "ok"
}
```

### `GET /api/info`

Return useful non-secret runtime information, for example:

```json
{
  "model": "yolo26n",
  "backend": "openvino",
  "device": "cpu",
  "imgsz": 640
}
```

### `POST /api/detect`

Accept one image.

Preferred request encoding for the first release:

```text
multipart/form-data
```

with an uploaded JPEG or PNG file.

Return JSON only.

Do not return a server-rendered annotated image as the primary API result.

---

## 10. Detection Response Contract

Use a predictable response structure.

Example:

```json
{
  "image": {
    "width": 640,
    "height": 480
  },
  "model": "yolo26n",
  "device": "cpu",
  "inference_ms": 84.3,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "box": {
        "x1": 120.4,
        "y1": 54.8,
        "x2": 410.2,
        "y2": 470.1
      }
    }
  ]
}
```

Bounding-box coordinates must refer to the uploaded image coordinate system.

Validate:

```text
0 <= x1 < x2 <= image width
0 <= y1 < y2 <= image height
```

Confidence values should be JSON numbers in the range `[0, 1]`.

Do not leak raw tensor objects or NumPy-specific types into JSON.

---

## 11. Browser Client

Use plain browser APIs.

The page must:

- ask for webcam permission;
- show the local camera preview;
- capture frames without refreshing the page;
- send frames at a controlled rate;
- display returned detections;
- draw boxes over the preview;
- show class name and confidence;
- show basic request/inference timing;
- allow starting and stopping detection.

Preferred architecture:

```text
<video>       local camera
<canvas>      frame capture
<canvas>      detection overlay, or one overlay canvas
fetch()       HTTP request to /api/detect
```

Do not send a new frame while the previous inference request is still outstanding unless concurrency is intentionally bounded.

The simplest safe first implementation is:

```text
capture
-> send
-> await response
-> draw
-> schedule next capture
```

This naturally applies backpressure and avoids building an unbounded queue.

---

## 12. Browser Camera Security Constraint

Modern browsers normally allow webcam access only in a secure context.

`localhost` is generally acceptable for local development.

For access from another machine over a LAN, plain HTTP may not permit webcam access depending on browser/security context.

The README must explain this clearly.

For tomorrow's demo, prefer one of these paths:

1. browser and service on the same Windows machine using `localhost`;
2. HTTPS termination if another client device must supply the camera;
3. use an uploaded image fallback if browser camera permission cannot be established.

Do not discover this issue during the live demo.

---

## 13. WSL2 Requirements

The service must run correctly inside Ubuntu under WSL2.

Keep project files inside the Linux filesystem where practical, for example:

```bash
~/work/yolo-web-demo
```

rather than relying on `/mnt/c/...`.

The server should run with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Document how to access it from the Windows host.

Do not make complicated LAN port forwarding mandatory for the basic demo.

The primary WSL2 acceptance path is:

```text
Windows browser -> localhost:8000 -> WSL2 FastAPI
```

If this is not supported by the actual host configuration, document the minimal tested alternative.

---

## 14. Dependency Policy

Prefer a short dependency list.

Expected dependencies include:

```text
ultralytics==8.4.157
fastapi
uvicorn[standard]
python-multipart
pillow
```

OpenCV may already arrive through Ultralytics dependencies; add explicit dependencies only when the application imports them directly.

For OpenVINO support, prefer the Ultralytics-supported installation mechanism:

```bash
pip install "ultralytics[export-openvino]==8.4.157"
```

Verify the exact extras syntax before finalizing `requirements.txt`.

If dependency resolution requires a slightly different form, use a tested equivalent and document it.

Do not add CUDA wheels.

Do not add large unrelated frameworks.

Do not add a database.

Do not add Redis.

Do not add Docker unless it materially improves the demo and remains optional.

---

## 15. Configuration

Configuration should be simple and environment-variable friendly.

Suggested settings:

```text
YOLO_MODEL=yolo26n.pt
YOLO_BACKEND=auto
YOLO_IMGSZ=640
YOLO_CONF=0.25
YOLO_MAX_DETECTIONS=100
HOST=0.0.0.0
PORT=8000
```

`YOLO_BACKEND=auto` may choose an already-exported OpenVINO model when present, otherwise use PyTorch CPU.

Do not silently use a GPU if one appears later.

CPU must remain the enforced device for this project unless the human explicitly changes the requirement.

---

## 16. Concurrency and CPU Protection

The service is a demo, not a high-throughput inference cluster.

Protect the machine from accidental request floods.

At minimum:

- do not create one model instance per request;
- avoid unlimited concurrent inference;
- prefer a small semaphore or serialized inference path if Ultralytics/backend thread behavior is uncertain;
- bound image upload size;
- reject malformed image content;
- do not retain uploaded images after inference unless explicitly requested.

Avoid premature multi-worker deployment.

Multiple Uvicorn workers would load multiple copies of the model and can increase memory/CPU contention.

Start with one process.

Benchmark before adding workers.

---

## 17. Performance Measurement

Performance claims require measurement.

Create a benchmark path or script that reports at least:

- model/backend;
- input resolution;
- CPU logical-core count if easily available;
- warm-up iterations;
- measured inference iterations;
- mean inference latency;
- median inference latency;
- approximate inference FPS;
- end-to-end HTTP latency if measured separately.

Warm up the model before timing.

Do not infer performance from Ultralytics marketing benchmarks.

Do not claim 2-5 FPS unless measured on the actual demo machine.

The browser must not attempt a fixed FPS higher than the server can sustain.

---

## 18. Tests

Tests are evidence.

A skipped test is not a passing test.

A test not run is not evidence.

Minimum automated tests:

### Health

`GET /health` returns HTTP 200 and expected status.

### Info

`GET /api/info` reports CPU as the device.

### Invalid upload

`POST /api/detect` rejects invalid image data cleanly.

### Valid detection contract

Using a small local test image, verify that:

- HTTP status is 200;
- response is valid JSON;
- `detections` exists;
- returned boxes, if any, have valid numeric coordinates;
- confidence is within range;
- device reports CPU.

Tests should not require a webcam.

Do not make real external URLs mandatory during tests.

A local fixture image is preferred.

---

## 19. Manual Demo Verification

Before calling the project demo-ready, manually verify:

```text
[ ] Clean environment installs successfully
[ ] Server starts without CUDA/NVIDIA
[ ] YOLO weights are available/download correctly
[ ] /health works
[ ] /api/info says CPU
[ ] Static web page loads
[ ] Browser requests camera access
[ ] Webcam preview appears
[ ] Frame is sent to server
[ ] Detection JSON returns
[ ] Bounding boxes align with visible objects
[ ] "person" detection works on a person when visible
[ ] Start/stop controls work
[ ] Browser remains responsive
[ ] No runaway request queue appears
[ ] Server survives at least several minutes of demo use
[ ] Measured latency/FPS is displayed or documented
```

Record any failure honestly.

---

## 20. Demo Reliability Features

Prefer boring reliability over ambitious functionality.

Useful safeguards:

- disable the Start button while already running;
- cancel the loop cleanly on Stop;
- display HTTP/server errors in the UI;
- show "no detections" rather than treating it as failure;
- set a reasonable client request timeout;
- prevent multiple overlapping detection loops;
- include a static image upload fallback;
- include a sample image test path if practical.

Do not add object tracking before basic detection is stable.

Do not add video recording.

Do not add authentication for the local demo unless external exposure requires it.

---

## 21. Security

This project accepts untrusted image uploads.

Apply basic safeguards:

- limit request body size;
- accept only expected image formats;
- decode images defensively;
- do not evaluate user-supplied code;
- do not accept arbitrary filesystem paths;
- do not expose shell commands;
- do not store uploads unnecessarily;
- do not include host environment variables in API responses;
- do not expose secrets in logs.

If the server is bound to `0.0.0.0`, explain that it may be reachable from other interfaces depending on firewall/network configuration.

For the demo, localhost-only access is preferable unless remote access is an explicit requirement.

---

## 22. Logging

Startup logs must report:

- Ultralytics version;
- model name/path;
- backend;
- CPU device;
- inference image size;
- confidence threshold;
- listening address.

Per-request logs may report:

- request ID;
- decode time;
- inference time;
- number of detections;
- total request time.

Do not log image contents.

Do not produce excessive per-frame debug output by default.

---

## 23. Documentation

`README.md` must contain a copy-paste quickstart.

At minimum document:

### Native Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### WSL2

Document the same Linux commands and the tested Windows browser URL.

### OpenVINO optimization

Document export as a separate optional/optimization step.

### Demo

Explain exactly:

1. start server;
2. open page;
3. permit camera;
4. press Start Detection;
5. stand in view;
6. observe person/object detections;
7. observe measured latency.

Documentation must distinguish:

- implemented;
- optional;
- experimental;
- not implemented.

---

## 24. Forbidden Actions

Do not:

- train a YOLO model;
- create a custom dataset;
- require a GPU;
- install CUDA as a requirement;
- use TensorRT;
- replace Ultralytics as the principal model interface;
- switch the project to YOLO11 merely because examples are easier to find;
- introduce a large frontend framework;
- introduce a database;
- implement user accounts;
- implement cloud deployment unless separately requested;
- introduce Kubernetes;
- add message queues;
- turn the API into an OpenAI API compatibility layer;
- store webcam frames by default;
- add unrelated features;
- claim benchmarks that were not measured;
- claim tests passed when they were skipped or not run;
- claim remote webcam access works over plain HTTP without testing browser security behavior.

---

## 25. Coding Style

Prefer readable code over abstraction.

Use:

- type hints for public Python functions;
- small functions;
- clear detector/API separation;
- Pydantic models for structured API responses where useful;
- explicit error handling around image decoding and inference;
- comments only where behavior is non-obvious.

Avoid:

- unnecessary classes;
- dependency injection frameworks;
- metaprogramming;
- generic plugin systems;
- speculative extensibility;
- duplicated inference code.

---

## 26. Git Workflow

For implementation work:

1. inspect repository state;
2. read this `AGENTS.md`;
3. start from current main/default branch;
4. create a feature branch;
5. implement one bounded task;
6. run relevant tests;
7. inspect the diff;
8. commit only related files;
9. report exact evidence;
10. do not merge unless explicitly instructed.

A reasonable first branch:

```text
feature/cpu-yolo-web-demo
```

Do not rewrite unrelated repository history.

---

## 27. Definition of Done for Initial Demo

The first release is done only when all of the following are true:

1. The project installs on the target Linux/WSL2 environment.
2. No GPU is required.
3. Ultralytics YOLO26n pretrained detection is used.
4. The service loads the model once.
5. `/api/detect` accepts a browser-produced image.
6. The endpoint returns structured JSON detections.
7. The HTML page obtains webcam images.
8. The page sends frames with controlled backpressure.
9. Bounding boxes and labels are rendered correctly.
10. A visible person can be detected using pretrained classes.
11. The UI displays measured timing information.
12. CPU execution is visible in `/api/info` or logs.
13. Automated API tests pass.
14. A manual webcam smoke test passes.
15. README contains tested startup instructions.
16. Known limitations are documented honestly.

---

## 28. First Implementation Work Order

Unless the repository already contains equivalent functionality, perform the following first task.

### Goal

Produce the smallest complete CPU-only browser-to-YOLO detection demo.

### Required implementation

- create FastAPI application;
- load `yolo26n.pt` through Ultralytics;
- enforce CPU inference;
- implement `/health`;
- implement `/api/info`;
- implement `/api/detect`;
- serve a minimal HTML page;
- capture webcam using HTML5;
- send JPEG frames using `fetch`;
- draw returned boxes;
- serialize requests rather than queueing them;
- show inference/request latency;
- add static image upload fallback;
- write setup documentation;
- add focused API tests.

### Non-goals

- OpenVINO optimization is not required in this first task if it delays a working end-to-end path;
- no tracking;
- no training;
- no authentication;
- no database;
- no Docker requirement;
- no external deployment;
- no WebSocket requirement.

### Verification

Run the focused test suite.

Then run the service and perform a manual browser smoke test.

Report actual observed inference latency.

If OpenVINO can be added safely after the baseline works, do it in a separate commit or separate PR-sized task.

---

## 29. Second Work Order: CPU Optimization

Only perform this after the initial demo works.

### Goal

Compare plain Ultralytics PyTorch CPU inference against Ultralytics-exported OpenVINO CPU inference.

### Required steps

1. export `yolo26n.pt` through Ultralytics to OpenVINO;
2. load the exported model through Ultralytics;
3. ensure execution remains CPU-only;
4. benchmark both backends using the same images and `imgsz`;
5. compare warm inference latency;
6. make OpenVINO the default only if it is stable and demonstrably beneficial on the target system;
7. preserve an easy fallback to the normal `.pt` model.

### Evidence

Report a table such as:

```text
Backend        Median ms    Mean ms    Approx FPS
PyTorch CPU    ...          ...        ...
OpenVINO CPU   ...          ...        ...
```

Do not fabricate missing numbers.

---

## 30. Agent Final Report Format

Every implementation task must end with:

```markdown
## Agent Report

### Summary
- What was implemented.

### Branch
- Branch name.

### Commit
- Commit hash, if committed.

### Files changed
- File-by-file short explanation.

### Runtime
- OS/environment.
- Python version.
- Ultralytics version.
- Model.
- Backend.
- Confirmed inference device.

### Tests run
- Exact command.
- Exact result.

### Manual verification
- Exact steps performed.
- Result.

### Performance
- Input size.
- Backend.
- Measured inference latency.
- Measured end-to-end latency if available.
- Approximate sustainable request/FPS rate.

### Dependencies installed
- Exact relevant packages or setup actions.

### Known limitations
- Anything not verified or not implemented.

### Risks
- Any concern relevant to tomorrow's demo.

### Recommended next action
- One narrow next task.
```

Never use phrases such as "all tests passed" unless all relevant requested tests actually ran and passed.

---

## 31. Strategic Escalation Rules

Stop implementation and report to the strategic model/human when:

- CPU-only operation cannot be achieved;
- official YOLO26 weights fail to load;
- Ultralytics compatibility requires changing the stated architecture;
- browser webcam use requires a different security/deployment model;
- a dependency conflict cannot be fixed without major changes;
- measured performance is too slow for an interactive demo;
- the implementation would require training;
- requested behavior would require a GPU;
- a proposed optimization materially increases demo fragility.

Do not hide these problems behind workarounds.

---

## 32. Guiding Principle

The purpose of this project is not to build the most sophisticated YOLO service.

The purpose is to produce a **credible, visible, reproducible CPU-only Ultralytics YOLO demonstration** that works reliably in front of another person.

Prefer:

```text
working demo
+ measured CPU inference
+ clear architecture
+ honest limitations
```

over:

```text
more features
+ more abstractions
+ more dependencies
+ less certainty
```
