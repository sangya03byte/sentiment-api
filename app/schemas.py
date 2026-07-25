from pydantic import BaseModel, Field


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
