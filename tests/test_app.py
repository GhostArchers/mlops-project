from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["message"] == "Barrier Prediction API is running"

def test_predict():
    res = client.post("/predict", json={
        "Area": 100,
        "Sensing Range": 10,
        "Transmission Range": 20,
        "Number of Sensor nodes": 50
    })
    assert res.status_code == 200
    assert "predicted_barriers" in res.json()

def test_ui_loads():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]