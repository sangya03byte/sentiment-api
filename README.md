# ML Sentiment Analysis API

A machine learning-powered REST API that analyses text sentiment using **Hugging Face DistilBERT**, with optional LLM-powered explanations through the **Claude API**.

## Features

- Sentiment classification (Positive / Negative).
- Single-text and batch predictions.
- Confidence score for predictions.
- Keyword-based fallback when the transformer model is unavailable.
- Optional AI-generated explanations using Claude API.
- Health check endpoint.
- Docker support.

## Tech Stack

**Python · FastAPI · Hugging Face Transformers · DistilBERT · Claude API · Docker · Pytest**

## Architecture

![Project Architecture](assets/architecture.png)

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Application health check |
| POST | `/predict` | Analyse a single text |
| POST | `/predict/batch` | Analyse multiple texts |

Interactive API documentation is available at `/docs` when the application is running.

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-api.git
cd sentiment-api
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_api_key
USE_TRANSFORMERS=true
```

**5. Start the application**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Visit `http://localhost:8080/docs` to explore the API.

## Docker

Build the image:

```bash
docker build -t sentiment-api .
```

Run the container:

```bash
docker run -p 8080:8080 sentiment-api
```

## Running Tests

```bash
pytest -v
```

## Future Enhancements

- GitHub Actions CI/CD automation.
- API authentication and rate limiting.
- Redis-based caching.
- Improved logging and monitoring.
- Prometheus and Grafana integration.
- Model evaluation and performance optimisation.

## Author

**Sangya Patel**

[LinkedIn](https://www.linkedin.com/in/sangyapatel18/) · [GitHub](https://github.com/sangya03byte)
