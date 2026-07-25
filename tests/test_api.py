import os
os.environ["USE_TRANSFORMERS"] = "false"
os.environ["ANTHROPIC_API_KEY"] = "sk-test-placeholder"

from fastapi.testclient import TestClient
from app.main import app
from app.model import SentimentModel

SentimentModel.load()
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_positive():
    r = client.post("/predict", json={"text": "I love this amazing product!", "include_explanation": False})
    assert r.status_code == 200
    assert r.json()["result"]["label"] == "POSITIVE"

def test_negative():
    r = client.post("/predict", json={"text": "Terrible broken awful waste.", "include_explanation": False})
    assert r.status_code == 200
    assert r.json()["result"]["label"] == "NEGATIVE"

def test_empty_rejected():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422

def test_batch():
    r = client.post("/predict/batch", json={"texts": ["Great!", "Awful."], "include_explanation": False})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2
