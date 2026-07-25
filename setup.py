import os

def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

write("app/__init__.py", "")

write(".gitignore", """.env
__pycache__/
*.pyc
.venv/
venv/
gcp-key.json
.pytest_cache/
""")

write(".dockerignore", """.env
.git
__pycache__
*.pyc
.pytest_cache
tests/
.venv
venv/
.github/
""")

write(".env", "ANTHROPIC_API_KEY=sk-ant-your-real-key-here\nUSE_TRANSFORMERS=true\n")

write("Dockerfile", """FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
""")

write("app/schemas.py", """from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    include_explanation: bool = Field(default=True)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)
    include_explanation: bool = Field(default=False)


class PredictionResult(BaseModel):
    text: str
    label: str
    score: float = Field(ge=0, le=1)
    explanation: str | None = None


class PredictResponse(BaseModel):
    result: PredictionResult
    model_name: str
    processing_time_ms: float


class BatchPredictResponse(BaseModel):
    results: list[PredictionResult]
    model_name: str
    total_texts: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
""")

write("app/model.py", """import logging
import os
import time

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
CONFIDENCE_THRESHOLD = 0.70
_USE_TRANSFORMERS = os.environ.get("USE_TRANSFORMERS", "true").lower() == "true"

_POSITIVE_WORDS = {
    "love", "great", "excellent", "amazing", "fantastic", "wonderful",
    "good", "best", "perfect", "awesome", "happy", "enjoyed", "fast",
    "reliable", "recommend", "easy", "helpful", "beautiful", "brilliant",
}
_NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "bad", "worst", "hate", "broken",
    "slow", "useless", "damaged", "poor", "disappointed", "waste",
    "defective", "missing", "wrong", "failed", "never", "problem", "issue",
}


def _keyword_predict(text: str) -> dict:
    words = set(text.lower().split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg or 1
    if pos >= neg:
        score = round(0.55 + 0.44 * (pos / total), 4)
        return {"label": "POSITIVE", "score": min(score, 0.99), "uncertain": score < CONFIDENCE_THRESHOLD}
    else:
        score = round(0.55 + 0.44 * (neg / total), 4)
        return {"label": "NEGATIVE", "score": min(score, 0.99), "uncertain": score < CONFIDENCE_THRESHOLD}


class SentimentModel:
    _pipeline = None
    _model_name = MODEL_NAME
    _loaded = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        start = time.time()
        if _USE_TRANSFORMERS:
            try:
                from transformers import pipeline
                cls._pipeline = pipeline("sentiment-analysis", model=cls._model_name)
                logger.info("HuggingFace pipeline loaded.")
            except Exception as e:
                logger.warning(f"HuggingFace failed ({e}). Using keyword fallback.")
                cls._pipeline = None
        else:
            cls._pipeline = None
        cls._loaded = True
        elapsed = (time.time() - start) * 1000
        logger.info(f"Model ready in {elapsed:.0f}ms")

    @classmethod
    def is_loaded(cls):
        return cls._loaded

    @classmethod
    def predict(cls, text: str) -> dict:
        if not cls._loaded:
            raise RuntimeError("Model not loaded.")
        if cls._pipeline is not None:
            r = cls._pipeline(text)[0]
            return {"label": r["label"], "score": round(r["score"], 4), "uncertain": r["score"] < CONFIDENCE_THRESHOLD}
        return _keyword_predict(text)

    @classmethod
    def predict_batch(cls, texts: list) -> list:
        if not cls._loaded:
            raise RuntimeError("Model not loaded.")
        if cls._pipeline is not None:
            return [
                {"label": r["label"], "score": round(r["score"], 4), "uncertain": r["score"] < CONFIDENCE_THRESHOLD}
                for r in cls._pipeline(texts)
            ]
        return [_keyword_predict(t) for t in texts]

    @classmethod
    def get_model_name(cls):
        return cls._model_name
""")

write("app/explainer.py", """import logging
import os
import anthropic

logger = logging.getLogger(__name__)
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def explain(text: str, label: str, score: float):
    try:
        client = get_client()
        word = "positive" if label == "POSITIVE" else "negative"
        confidence_pct = int(score * 100)
        prompt = (
            f"Customer review: {text}\\n\\n"
            f"Sentiment model result: {word.upper()} ({confidence_pct}% confidence).\\n\\n"
            f"In exactly 2 sentences:\\n"
            f"1. What is driving this sentiment?\\n"
            f"2. One concrete business action to take."
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except anthropic.RateLimitError:
        logger.warning("Claude rate limit hit.")
        return None
    except anthropic.APIConnectionError:
        logger.warning("Claude connection error.")
        return None
    except Exception as e:
        logger.error(f"Claude error: {e}")
        return None
""")

write("app/main.py", """import logging
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import SentimentModel
from app.explainer import explain
from app.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    PredictionResult, HealthResponse,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    logger.info("Starting up...")
    SentimentModel.load()
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Sentiment Analysis API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health_check():
    return HealthResponse(status="ok", model_loaded=SentimentModel.is_loaded())


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    start = time.time()
    try:
        pred = SentimentModel.predict(request.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    explanation = None
    if request.include_explanation:
        explanation = explain(request.text, pred["label"], pred["score"])
    return PredictResponse(
        result=PredictionResult(
            text=request.text,
            label=pred["label"],
            score=pred["score"],
            explanation=explanation,
        ),
        model_name=SentimentModel.get_model_name(),
        processing_time_ms=round((time.time() - start) * 1000, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(request: BatchPredictRequest):
    start = time.time()
    try:
        preds = SentimentModel.predict_batch(request.texts)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    results = []
    for text, pred in zip(request.texts, preds):
        explanation = explain(text, pred["label"], pred["score"]) if request.include_explanation else None
        results.append(PredictionResult(
            text=text, label=pred["label"], score=pred["score"], explanation=explanation,
        ))
    return BatchPredictResponse(
        results=results,
        model_name=SentimentModel.get_model_name(),
        total_texts=len(request.texts),
        processing_time_ms=round((time.time() - start) * 1000, 2),
    )
""")

write("tests/test_api.py", """import os
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
""")

print("\nAll files created successfully!")
print("\nStructure:")
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", "venv", ".pytest_cache"]]
    level = root.replace(".", "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for file in files:
        print(f"{indent}  {file}")
