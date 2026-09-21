"""POST /api/detect contract tests: validation, errors, and response shape."""


def test_invalid_image_rejected_cleanly(client):
    response = client.post(
        "/api/detect",
        files={"file": ("bad.jpg", b"this is not an image", "image/jpeg")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "image" in detail.lower()
    assert "Traceback" not in response.text


def test_empty_upload_rejected(client):
    response = client.post(
        "/api/detect",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_unsupported_format_rejected(client, sample_gif):
    response = client.post(
        "/api/detect",
        files={"file": ("anim.gif", sample_gif, "image/gif")},
    )
    assert response.status_code == 400
    assert "unsupported image format" in response.json()["detail"]


def test_oversized_upload_rejected(client):
    payload = b"\x00" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/detect",
        files={"file": ("big.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413


def test_missing_file_field_rejected(client):
    response = client.post("/api/detect")
    assert response.status_code == 422


def test_detect_returns_valid_contract(client, sample_jpeg):
    response = client.post(
        "/api/detect",
        files={"file": ("frame.jpg", sample_jpeg, "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["image"] == {"width": 320, "height": 240}
    assert payload["model"] == "yolo26n.pt"
    assert payload["device"] == "cpu"
    assert isinstance(payload["inference_ms"], (int, float))
    assert payload["inference_ms"] >= 0
    assert isinstance(payload["detections"], list)
    assert len(payload["detections"]) == 1

    det = payload["detections"][0]
    assert det["class_id"] == 0
    assert det["class_name"] == "person"
    assert isinstance(det["confidence"], (int, float))
    assert 0.0 <= det["confidence"] <= 1.0

    box = det["box"]
    for key in ("x1", "y1", "x2", "y2"):
        assert isinstance(box[key], (int, float)), f"box.{key} not numeric"
    assert 0 <= box["x1"] < box["x2"] <= 320
    assert 0 <= box["y1"] < box["y2"] <= 240


def test_png_accepted(client, sample_png):
    response = client.post(
        "/api/detect",
        files={"file": ("frame.png", sample_png, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["image"] == {"width": 320, "height": 240}


def test_no_detections_is_success(client, small_jpeg):
    response = client.post(
        "/api/detect",
        files={"file": ("small.jpg", small_jpeg, "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["detections"] == []
    assert payload["image"] == {"width": 64, "height": 48}
