"""Source-level guards for the browser client (app/static/app.js).

The project has no JavaScript runtime or build toolchain (no Node), so a
behavioral JS test is not possible here; the real-browser webcam acceptance
test is a separate work order. These lightweight checks pin the Work Order
001b fixes at source level: Stop must abort the in-flight request, stale
responses must be dropped before any draw, an intentional abort must not be
shown as an error, and the live rate must come from wall-clock time rather
than latency math.
"""

import re
from pathlib import Path

JS_PATH = Path(__file__).resolve().parents[1] / "app" / "static" / "app.js"


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def _function_source(name: str) -> str:
    # Top-level functions only: bodies are indented, so the first column-0
    # closing brace after the header ends the function.
    src = _js()
    start = src.index(f"function {name}(")
    end = src.index("\n}", start)
    return src[start : end + 2]


def test_stop_detection_aborts_inflight_request():
    stop = _function_source("stopDetection")
    assert "loopAbort.abort()" in stop
    # The stop flag must be set before the abort so a woken loop sees it.
    assert stop.index("running = false") < stop.index("loopAbort.abort()")


def test_start_detection_creates_fresh_loop_generation():
    start = _function_source("startDetection")
    assert "new AbortController()" in start
    assert "loopSeq" in start
    # A double click on Start must not be able to spawn a second loop.
    assert "startBtn.disabled" in start


def test_loop_drops_stale_responses_after_stop():
    loop = _function_source("detectionLoop")
    fetch = loop.index("await postImageToDetect(")
    draw = loop.index("drawDetections(")
    between = loop[fetch:draw]
    # A liveness check must separate the awaited response from the draw.
    assert "!alive()" in between


def test_stop_abort_is_not_displayed_as_error():
    loop = _function_source("detectionLoop")
    catch = loop.index("catch (err)")
    first_error = loop.index("showError(", catch)
    segment = loop[catch:first_error]
    assert "!alive()" in segment  # break out silently before any error display


def test_live_fps_uses_wall_clock_not_latency_math():
    assert "1000 / avgMs" not in _js()  # old latency-based estimate removed
    loop = _function_source("detectionLoop")
    assert "loopStartedAt" in loop
    assert re.search(r"samples\s*/\s*\(?\s*elapsedMs\s*/\s*1000", loop)
    assert "det/s" in loop
