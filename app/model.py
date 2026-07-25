import logging
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
