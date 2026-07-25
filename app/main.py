import logging
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
