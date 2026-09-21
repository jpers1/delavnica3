"""GET /api/info reports the runtime, with device fixed to CPU."""


def test_info_reports_cpu_device(client):
    response = client.get("/api/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["device"] == "cpu"
    assert payload["backend"] == "pytorch"
    assert payload["model"] == "yolo26n.pt"
    assert payload["imgsz"] == 640
    assert 0 < payload["conf"] <= 1


def test_info_contains_expected_fields(client):
    payload = client.get("/api/info").json()
    for key in ("model", "backend", "device", "imgsz", "conf", "ultralytics_version"):
        assert key in payload, f"missing field {key!r}"
