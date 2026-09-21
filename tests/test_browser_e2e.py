"""End-to-end browser test of the webcam detection client (Work Order 001c).

Drives the real page in Playwright Chromium against the real FastAPI backend
(real ``yolo26n.pt``, CPU inference) with a fake webcam: Chromium's fake-media
support fed by a repeating ``.y4m`` video generated with ffmpeg from the
Ultralytics ``zidane.jpg`` sample (which contains people).

This exercises the real browser code path end to end:

    getUserMedia -> <video> -> canvas -> JPEG blob -> fetch /api/detect
    -> real YOLO CPU inference -> JSON -> overlay canvas

It also covers: the Stop race with a deliberately delayed response (the
in-flight request is held in flight by a route and the Stop click is issued
while it is held), the Start -> Stop -> Start single-loop guarantee, and the
static-image fallback.

The test uses Playwright's *async* API on a persistent event loop (daemon
thread). Rationale: the Stop-race phase needs a non-blocking route handler
(``await asyncio.sleep`` inside ``route`` interception) so the driver stays
free to process the Stop click while a response is held. With the sync API a
route handler sleep blocks the dispatcher, which makes the deterministic
setup impossible.

Test-only prerequisites (deliberately not in requirements.txt):

    pip install playwright
    python -m playwright install chromium    (+ system deps)
    ffmpeg                                   (e.g. sudo apt-get install -y ffmpeg)

The test skips cleanly when any prerequisite is missing. It proves the
software data path in a real browser engine; it does NOT replace the
physical-camera demo check required by AGENTS.md.
"""

import asyncio
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

try:
    from playwright.async_api import async_playwright
except ImportError:  # test-only dependency
    async_playwright = None

ROOT = Path(__file__).resolve().parents[1]
PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"


def _chromium_binary():
    cache = Path.home() / ".cache" / "ms-playwright"
    for pat in (
        "chromium-*/chrome-linux64/chrome",
        "chromium-*/chrome-linux/chrome",
    ):
        hits = sorted(cache.glob(pat))
        if hits:
            return hits[-1]
    return None


def _zidane():
    try:
        import ultralytics

        cand = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
        if cand.exists():
            return cand
    except Exception:
        pass
    hits = sorted(ROOT.rglob("zidane.jpg"))
    return hits[0] if hits else None


def _overlay_pixel_count(canvas_id: str) -> str:
    """JS function expression: count of non-transparent pixels on a canvas."""
    return f"""
    () => {{
      const c = document.getElementById("{canvas_id}");
      if (!c.width || !c.height) return 0;
      const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      let n = 0;
      for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
      return n;
    }}
    """


def _missing():
    missing = []
    if async_playwright is None:
        missing.append("playwright (pip install playwright)")
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg (e.g. sudo apt-get install -y ffmpeg)")
    if _chromium_binary() is None:
        missing.append("chromium (python -m playwright install chromium)")
    return "; ".join(missing)


pytestmark = pytest.mark.skipif(
    bool(_missing()), reason="browser e2e prerequisites missing: " + _missing()
)


# --------------------------------------------------------------------------
# environment: real backend + Playwright browser on a persistent async loop
# --------------------------------------------------------------------------


class PWEnv:
    """Playwright (async) running on a daemon thread with one shared loop."""

    def __init__(self, fake_video: Path):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, daemon=True, name="pw-loop"
        )
        self.thread.start()
        self._pw = None
        self.browser = None
        self.page = None
        self.detect_responses = []
        self.detect_requests = []
        self.chromium_version = asyncio.run_coroutine_threadsafe(
            self._setup(fake_video), self.loop
        ).result(timeout=180)

    async def _setup(self, fake_video: Path):
        self._pw = await async_playwright().start()
        self.browser = await self._pw.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-video-capture={fake_video}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        ctx = await self.browser.new_context(
            viewport={"width": 1000, "height": 900}
        )
        self.page = await ctx.new_page()
        self.page.on("response", self._on_response)
        self.page.on("request", self._on_request)
        return self.browser.version

    async def _on_response(self, resp):
        if resp.url.endswith("/api/detect"):
            try:
                body = await resp.json()
            except Exception:
                body = None
            self.detect_responses.append({"status": resp.status, "json": body})

    def _on_request(self, req):
        if req.url.endswith("/api/detect") and req.method == "POST":
            self.detect_requests.append(req.url)

    def run(self, coro, timeout=120):
        """Run an async test phase on the shared loop and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=timeout)

    def close(self):
        async def _close():
            try:
                if self.page:
                    await self.page.close()
            finally:
                if self.browser:
                    await self.browser.close()
                if self._pw:
                    await self._pw.stop()

        try:
            asyncio.run_coroutine_threadsafe(_close(), self.loop).result(timeout=30)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=10)


@pytest.fixture(scope="module")
def fake_video(tmp_path_factory):
    src = _zidane()
    assert src is not None, "zidane.jpg sample not found (expected in ultralytics assets)"
    out = tmp_path_factory.mktemp("fakecam") / "test-camera.y4m"
    # Repeating freeze-frame of zidane.jpg, 640x480 @ 15 fps (YUV4MPEG2 / .y4m).
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-loop", "1", "-i", str(src),
            "-t", "10", "-r", "15",
            "-vf", "scale=640:480:force_original_aspect_ratio=decrease,"
                   "pad=640:480:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-f", "yuv4mpegpipe", str(out),
        ],
        check=True,
    )
    print(f"[e2e] fake webcam video: {out}")
    return out


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    log = tmp_path_factory.mktemp("serverlog") / "uvicorn.log"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT,
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 120
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early:\n{log.read_text()[-2000:]}")
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=1) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.5)
    assert ready, f"server did not become healthy within 120s (log: {log})"
    print(f"[e2e] real backend on {BASE} (real yolo26n.pt, CPU)")
    yield BASE
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def pw(server, fake_video):
    env = PWEnv(fake_video)
    print(f"[e2e] chromium version: {env.chromium_version} (headless, fake media)")
    yield env
    env.close()


# --------------------------------------------------------------------------
# tests (sequential; they share the page and its loop state on purpose)
# --------------------------------------------------------------------------


def test_page_loads_and_info(pw):
    async def phase():
        page = pw.page
        await page.goto(BASE, wait_until="load")
        assert await page.title() == "YOLO CPU Web Demo"
        # /api/info is displayed, including the CPU device.
        await page.wait_for_function(
            "() => document.getElementById('info').textContent.includes('device: cpu')",
            timeout=15000,
        )
        info = await page.locator("#info").text_content()
        assert "yolo26n.pt" in info
        print(f"[e2e] info bar: {info}")
        # Start Detection is clickable.
        assert await page.locator("#start-btn").is_enabled()

    pw.run(phase())


def test_webcam_detection_loop(pw, tmp_path):
    async def phase():
        page = pw.page
        pw.detect_responses.clear()
        pw.detect_requests.clear()

        await page.click("#start-btn")

        # getUserMedia succeeded and the video element is playing the fake camera.
        await page.wait_for_function(
            "() => { const v = document.getElementById('video');"
            " return v.videoWidth > 0 && v.videoHeight > 0; }",
            timeout=20000,
        )
        dims = await page.evaluate(
            "() => [document.getElementById('video').videoWidth,"
            " document.getElementById('video').videoHeight]"
        )
        print(f"[e2e] fake camera video dimensions: {dims}")
        assert dims == [640, 480]

        # Stats show request latency, server inference latency, and the live rate.
        await page.wait_for_function(
            "() => { const t = document.getElementById('stats').textContent;"
            " return t.includes('request') && t.includes('server')"
            " && t.includes('det/s'); }",
            timeout=60000,
        )
        stats = await page.locator("#stats").text_content()
        print(f"[e2e] stats line: {stats}")

        # Real YOLO must produce person detection(s) from the fake-camera person video.
        def detected_classes():
            out = set()
            for d in pw.detect_responses:
                if d["json"]:
                    out.update(x["class_name"] for x in d["json"].get("detections", []))
            return out

        deadline = time.time() + 60
        while time.time() < deadline and "person" not in detected_classes():
            await page.wait_for_timeout(300)
        classes = sorted(detected_classes())
        print(f"[e2e] detected classes: {classes}")
        assert "person" in classes

        # Every response came from the real backend, on CPU.
        deadline = time.time() + 60
        while time.time() < deadline and len(pw.detect_responses) < 3:
            await page.wait_for_timeout(200)
        assert len(pw.detect_responses) >= 3, "too few /api/detect responses"
        assert all(d["status"] == 200 for d in pw.detect_responses)
        assert all(d["json"]["device"] == "cpu" for d in pw.detect_responses)
        assert all(d["json"]["model"] == "yolo26n.pt" for d in pw.detect_responses)

        # The overlay canvas actually contains drawn bounding boxes.
        assert await page.evaluate(_overlay_pixel_count("overlay")) > 0
        print(f"[e2e] /api/detect POSTs observed: {len(pw.detect_requests)}")
        shot = tmp_path / "webcam_boxes.png"
        await page.screenshot(path=str(shot))
        print(f"[e2e] screenshot with boxes: {shot}")

    pw.run(phase())


def test_stop_drops_stale_response(pw):
    async def phase():
        page = pw.page
        assert await page.evaluate("running") is True, "loop should still be running"

        # Route /api/detect: fetch the real server response, then HOLD it for
        # 2 s (non-blocking, so the driver can process the Stop click while
        # the response is in flight). One response is thereby guaranteed to
        # land AFTER Stop.
        held = {"in": 0, "out": 0}

        async def hold_route(route):
            response = await route.fetch()
            held["in"] += 1
            await asyncio.sleep(2.0)
            try:
                await route.fulfill(response=response)
            except Exception:
                pass  # request was aborted by Stop; the fulfill is a no-op
            finally:
                held["out"] += 1

        await page.route("**/api/detect**", hold_route)
        deadline = time.time() + 30
        while time.time() < deadline and held["in"] < 1:
            await page.wait_for_timeout(100)
        assert held["in"] >= 1, "no request reached the holding route"

        # Stop while the response is held in flight.
        await page.click("#stop-btn")

        # Stopped state, immediately.
        assert await page.locator("#stats").text_content() == "Stopped."
        await page.wait_for_function(
            "document.getElementById('video').srcObject === null", timeout=5000
        )
        assert await page.evaluate("stream") is None
        assert await page.evaluate("running") is False
        assert await page.locator("#start-btn").is_enabled()
        assert await page.locator("#stop-btn").is_disabled()
        assert await page.locator("#error").is_hidden(), "abort shown as error"
        assert await page.evaluate(_overlay_pixel_count("overlay")) == 0

        requests_after_stop = len(pw.detect_requests)
        # Wait for the held (stale) response to be released and give it time
        # to be (wrongly) processed.
        deadline = time.time() + 20
        while time.time() < deadline and held["out"] < 1:
            await page.wait_for_timeout(100)
        assert held["out"] >= 1, "held response was never released"
        await page.wait_for_timeout(1000)

        # The stale response must not redraw boxes, must not overwrite the
        # status, and the stopped loop must not send further requests.
        assert await page.locator("#stats").text_content() == "Stopped."
        assert await page.evaluate(_overlay_pixel_count("overlay")) == 0
        assert len(pw.detect_requests) == requests_after_stop, "new request after Stop"
        print("[e2e] stop test: in-flight request aborted at Stop; stale response "
              "dropped, status stayed 'Stopped.', no new requests")
        await page.unroute("**/api/detect**")

    pw.run(phase())


def test_start_stop_start_single_loop(pw):
    async def phase():
        page = pw.page

        # Count how many times the (top-level) detectionLoop function is launched.
        await page.evaluate(
            """
            () => {
              window.__loops = 0;
              window.__origLoop = window.detectionLoop;
              window.detectionLoop = function (...args) {
                window.__loops += 1;
                return window.__origLoop.apply(this, args);
              };
            }
            """
        )

        await page.click("#start-btn")
        await page.wait_for_function(
            "() => document.getElementById('video').videoWidth > 0", timeout=20000
        )
        await page.wait_for_function(
            "() => document.getElementById('stats').textContent.includes('det/s')",
            timeout=60000,
        )
        before = len(pw.detect_responses)
        deadline = time.time() + 60
        while time.time() < deadline and len(pw.detect_responses) - before < 3:
            await page.wait_for_timeout(300)
        assert len(pw.detect_responses) - before >= 3, "restart produced no responses"

        loops = await page.evaluate("window.__loops")
        assert loops == 1, f"expected exactly 1 detection loop after restart, saw {loops}"
        assert await page.evaluate("running") is True
        print("[e2e] start->stop->start: exactly one detection loop operating")

        # Stop still works cleanly after the restart.
        await page.click("#stop-btn")
        assert await page.locator("#stats").text_content() == "Stopped."
        await page.wait_for_function(
            "document.getElementById('video').srcObject === null", timeout=5000
        )
        assert await page.evaluate(_overlay_pixel_count("overlay")) == 0
        # Restore the original function.
        await page.evaluate("() => { window.detectionLoop = window.__origLoop; }")

    pw.run(phase())


def test_static_image_fallback(pw, tmp_path):
    async def phase():
        page = pw.page
        pw.detect_responses.clear()

        src = _zidane()
        await page.set_input_files("#file", str(src))
        assert await page.locator("#upload-btn").is_enabled()
        await page.click("#upload-btn")

        await page.wait_for_function(
            "() => { const t = document.getElementById('upload-stats').textContent;"
            " return t.length > 0 && !t.startsWith('Uploading'); }",
            timeout=60000,
        )
        stats = await page.locator("#upload-stats").text_content()
        print(f"[e2e] upload stats: {stats}")

        assert len(pw.detect_responses) == 1, "upload did not hit /api/detect exactly once"
        resp = pw.detect_responses[0]
        assert resp["status"] == 200
        assert resp["json"]["device"] == "cpu"
        names = [x["class_name"] for x in resp["json"]["detections"]]
        print(f"[e2e] upload detections: {names}")
        assert "person" in names

        assert await page.evaluate(_overlay_pixel_count("upload-overlay")) > 0
        shot = tmp_path / "upload_overlay.png"
        await page.screenshot(path=str(shot))
        print(f"[e2e] screenshot with upload overlay: {shot}")

    pw.run(phase())
