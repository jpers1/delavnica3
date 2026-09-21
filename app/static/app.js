"use strict";

/* CPU-only YOLO web demo client.
 *
 * Backpressure model: one capture -> one HTTP request -> await response ->
 * draw -> wait configured delay -> next capture. A new frame is never sent
 * while a previous inference request is still outstanding, so no unbounded
 * queue can build up against the CPU-only backend.
 *
 * Stop semantics: Stop sets `running` to false, bumps `loopSeq` (which
 * invalidates the current loop generation) and aborts the in-flight webcam
 * request through its AbortController. The loop re-checks `alive()` after
 * every await, so a stale response can never redraw boxes or overwrite the
 * stopped status, and an intentional abort is never displayed as an error.
 * A Start that is still coming up when Stop is pressed is likewise
 * invalidated: it discards its camera stream instead of starting a loop.
 */

const $ = (id) => document.getElementById(id);

const video = $("video");
const overlay = $("overlay");
const emptyNote = $("video-empty");
const startBtn = $("start-btn");
const stopBtn = $("stop-btn");
const delayInput = $("delay");
const stats = $("stats");
const errorBox = $("error");
const infoBox = $("info");

const uploadBtn = $("upload-btn");
const fileInput = $("file");
const uploadImg = $("upload-img");
const uploadOverlay = $("upload-overlay");
const uploadStats = $("upload-stats");

const REQUEST_TIMEOUT_MS = 15000;

let stream = null;
let running = false; // detection loop active
let loopSeq = 0; // generation counter; bumped on start/stop to invalidate loops
let loopAbort = null; // AbortController for the in-flight webcam request
let loopPromise = null;

/* ---------- small helpers ---------- */

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function frameDelayMs() {
  const value = parseInt(delayInput.value, 10);
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

function showError(message) {
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

async function loadInfo() {
  try {
    const res = await fetch("/api/info");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const info = await res.json();
    infoBox.textContent =
      `${info.model} · ${info.backend} · device: ${info.device}` +
      ` · imgsz: ${info.imgsz} · conf: ${info.conf}`;
  } catch {
    infoBox.textContent = "model info unavailable (is the server running?)";
  }
}

async function postImageToDetect(blob, filename, externalSignal = null) {
  const formData = new FormData();
  formData.append("file", blob, filename);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let onExternalAbort = null;
  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort();
    } else {
      onExternalAbort = () => controller.abort();
      externalSignal.addEventListener("abort", onExternalAbort, { once: true });
    }
  }
  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    if (!res.ok) {
      let detail = "HTTP " + res.status;
      try {
        const body = await res.json();
        if (body.detail) detail += ": " + body.detail;
      } catch {
        /* non-JSON error body */
      }
      throw new Error(detail);
    }
    return res.json();
  } finally {
    clearTimeout(timeout);
    if (externalSignal && onExternalAbort) {
      externalSignal.removeEventListener("abort", onExternalAbort);
    }
  }
}

/* ---------- overlay drawing ---------- */

function drawDetections(canvas, detections, imageWidth, imageHeight) {
  // The canvas' internal size is set to the frame's native size, so box
  // coordinates (in uploaded-image pixels) map 1:1; CSS stretches the canvas
  // over the displayed video, which shares the same aspect ratio.
  canvas.width = imageWidth;
  canvas.height = imageHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  ctx.font = "14px sans-serif";
  for (const det of detections) {
    const { box } = det;
    const w = box.x2 - box.x1;
    const h = box.y2 - box.y1;
    ctx.strokeStyle = "#00e676";
    ctx.strokeRect(box.x1, box.y1, w, h);
    const label = `${det.class_name} ${det.confidence.toFixed(2)}`;
    const textW = ctx.measureText(label).width;
    const labelTop = box.y1 > 20 ? box.y1 - 18 : box.y1 + 2;
    ctx.fillStyle = "#00e676";
    ctx.fillRect(box.x1, labelTop, textW + 8, 17);
    ctx.fillStyle = "#04140b";
    ctx.fillText(label, box.x1 + 4, labelTop + 13);
  }
}

/* ---------- webcam capture + detection loop ---------- */

function captureFrame() {
  // Draw the current video frame at native resolution; encode as JPEG.
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) =>
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85)
  );
}

async function detectionLoop(mySeq, signal) {
  let samples = 0;
  const loopStartedAt = performance.now();
  const alive = () => running && loopSeq === mySeq;
  while (alive()) {
    if (!video.videoWidth || !video.videoHeight) {
      await sleep(100); // camera not ready yet
      continue;
    }
    const frame = await captureFrame();
    if (!frame || !alive()) break;

    const started = performance.now();
    let response;
    try {
      response = await postImageToDetect(frame, "frame.jpg", signal);
    } catch (err) {
      if (!alive()) break; // Stop was pressed: the abort is intentional, not an error
      if (err && err.name === "AbortError") {
        showError("Detection request timed out.");
      } else {
        showError("Detection request failed: " + err.message);
      }
      await sleep(1000); // pause instead of hammering a broken backend
      continue;
    }
    if (!alive()) break; // response arrived after Stop; drop it, draw nothing

    const requestMs = performance.now() - started;
    samples += 1;
    clearError();

    drawDetections(
      overlay,
      response.detections,
      response.image.width,
      response.image.height
    );

    // Live rate = completed detections / wall-clock time since Start.
    // Deliberately not derived from request or model latency.
    const elapsedMs = performance.now() - loopStartedAt;
    const liveFps = elapsedMs > 0 ? samples / (elapsedMs / 1000) : 0;
    const count = response.detections.length;
    stats.textContent =
      (count > 0 ? `${count} object(s) detected` : "No detections") +
      ` · request ${requestMs.toFixed(0)} ms` +
      ` · server ${Number(response.inference_ms).toFixed(0)} ms` +
      ` · live ${liveFps.toFixed(1)} det/s`;

    // Backpressure: wait for the configured delay before the next frame.
    const delay = frameDelayMs();
    if (delay > 0) await sleep(delay);
  }
}

async function startDetection() {
  if (running || startBtn.disabled) return; // already starting or running
  startBtn.disabled = true; // block double-start while the camera comes up
  clearError();
  const seqBefore = loopSeq; // Stop bumps loopSeq; that marks an interrupted start
  let newStream;
  try {
    newStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    startBtn.disabled = false;
    showError(
      "Camera access failed: " + err.message +
      " — try the static image fallback below."
    );
    return;
  }
  if (loopSeq !== seqBefore) {
    // Stop was pressed while the permission prompt was up: discard the camera.
    for (const track of newStream.getTracks()) track.stop();
    return; // Stop already restored the stopped UI state
  }

  stream = newStream;
  video.srcObject = stream;
  stopBtn.disabled = false; // camera live: Stop can cancel startup or the loop
  await new Promise((resolve) => {
    if (video.readyState >= 2) resolve();
    else video.addEventListener("loadeddata", resolve, { once: true });
  });
  if (loopSeq !== seqBefore || stream !== newStream) {
    // Stop was pressed while the first frame was loading: discard the camera.
    for (const track of newStream.getTracks()) track.stop();
    if (stream === newStream) {
      stream = null;
      video.srcObject = null;
      stopBtn.disabled = true;
      startBtn.disabled = false;
    }
    return;
  }

  emptyNote.hidden = true;

  running = true;
  loopSeq += 1;
  loopAbort = new AbortController();
  stats.textContent = "Detecting\u2026";
  loopPromise = detectionLoop(loopSeq, loopAbort.signal);
}

async function stopDetection() {
  running = false;
  loopSeq += 1; // invalidate the current loop generation
  if (loopAbort) {
    loopAbort.abort(); // cancel the in-flight webcam request (if any)
    loopAbort = null;
  }
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
    video.srcObject = null;
  }
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  stats.textContent = "Stopped.";
  const exitingLoop = loopPromise;
  loopPromise = null;
  if (exitingLoop) {
    await exitingLoop; // wait until the invalidated loop has fully exited
  }
}

/* ---------- static image fallback ---------- */

fileInput.addEventListener("change", () => {
  uploadBtn.disabled = fileInput.files.length === 0;
});

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  clearError();
  uploadStats.textContent = "Uploading\u2026";
  uploadImg.hidden = true;
  uploadOverlay.hidden = true;

  const url = URL.createObjectURL(file);
  try {
    const natural = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
      img.onerror = () => reject(new Error("could not load the selected image"));
      img.src = url;
    });

    const started = performance.now();
    const data = await postImageToDetect(file, file.name || "upload.jpg");
    const requestMs = performance.now() - started;

    uploadImg.src = url;
    uploadImg.hidden = false;
    uploadOverlay.hidden = false;
    drawDetections(
      uploadOverlay,
      data.detections,
      data.image.width,
      data.image.height
    );
    const count = data.detections.length;
    uploadStats.textContent =
      (count > 0 ? `${count} object(s)` : "No detections") +
      ` · request ${requestMs.toFixed(0)} ms` +
      ` · server ${Number(data.inference_ms).toFixed(0)} ms`;
  } catch (err) {
    showError("Image detection failed: " + err.message);
    uploadStats.textContent = "";
  }
});

/* ---------- wiring ---------- */

startBtn.addEventListener("click", startDetection);
stopBtn.addEventListener("click", stopDetection);
loadInfo();
