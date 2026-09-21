"use strict";

/* CPU-only YOLO web demo client.
 *
 * Backpressure model: one capture -> one HTTP request -> await response ->
 * draw -> wait configured delay -> next capture. A new frame is never sent
 * while a previous inference request is still outstanding, so no unbounded
 * queue can build up against the CPU-only backend.
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

async function postImageToDetect(blob, filename) {
  const formData = new FormData();
  formData.append("file", blob, filename);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
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

async function detectionLoop() {
  let samples = 0;
  let totalRequestMs = 0;
  while (running) {
    if (!video.videoWidth || !video.videoHeight) {
      await sleep(100); // camera not ready yet
      continue;
    }
    const frame = await captureFrame();
    if (!frame || !running) break;

    const started = performance.now();
    let response;
    try {
      response = await postImageToDetect(frame, "frame.jpg");
    } catch (err) {
      if (!running) break;
      showError("Detection request failed: " + err.message);
      await sleep(1000); // pause instead of hammering a broken backend
      continue;
    }
    const requestMs = performance.now() - started;
    samples += 1;
    totalRequestMs += requestMs;
    clearError();

    drawDetections(
      overlay,
      response.detections,
      response.image.width,
      response.image.height
    );

    const avgMs = totalRequestMs / samples;
    const perSecond = avgMs > 0 ? (1000 / avgMs).toFixed(1) : "?";
    const count = response.detections.length;
    stats.textContent =
      (count > 0 ? `${count} object(s) detected` : "No detections") +
      ` · request ${requestMs.toFixed(0)} ms` +
      ` · server ${Number(response.inference_ms).toFixed(0)} ms` +
      ` · ~${perSecond} img/s`;

    // Backpressure: wait for the configured delay before the next frame.
    const delay = frameDelayMs();
    if (delay > 0) await sleep(delay);
  }
}

async function startDetection() {
  if (running) return;
  clearError();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    showError(
      "Camera access failed: " + err.message +
      " — try the static image fallback below."
    );
    return;
  }

  video.srcObject = stream;
  await new Promise((resolve) => {
    if (video.readyState >= 2) resolve();
    else video.onloadeddata = resolve;
  });
  emptyNote.hidden = true;

  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  stats.textContent = "Detecting\u2026";
  loopPromise = detectionLoop();
}

async function stopDetection() {
  running = false;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
    video.srcObject = null;
  }
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  stats.textContent = "Stopped.";
  if (loopPromise) {
    await loopPromise; // loop notices running=false and exits on its own
    loopPromise = null;
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
