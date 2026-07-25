import logging
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
            f"Customer review: {text}\n\n"
            f"Sentiment model result: {word.upper()} ({confidence_pct}% confidence).\n\n"
            f"In exactly 2 sentences:\n"
            f"1. What is driving this sentiment?\n"
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
