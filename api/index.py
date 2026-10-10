
import json
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# Enable CORS for all origins, methods, and headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Existing telemetry API
DATA_FILE = Path(__file__).parent / "data.json"


class MetricsRequest(BaseModel):
    regions: list[str]
    threshold_ms: float = Field(ge=0)


def load_telemetry():
    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="Unable to load telemetry data",
        )


@app.post("/")
def calculate_metrics(request: MetricsRequest):
    records = load_telemetry()
    results = {}

    for region in request.regions:
        matching = [
            record
            for record in records
            if record["region"].lower() == region.lower()
        ]

        if not matching:
            results[region] = {
                "avg_latency": None,
                "p95_latency": None,
                "avg_uptime": None,
                "breaches": 0,
            }
            continue

        latencies = [record["latency_ms"] for record in matching]
        uptimes = [record["uptime_pct"] for record in matching]

        results[region] = {
            "avg_latency": float(np.mean(latencies)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "avg_uptime": float(np.mean(uptimes)),
            "breaches": sum(
                latency > request.threshold_ms
                for latency in latencies
            ),
        }

    return {"regions": results}


# New sentiment analysis API
class SentimentRequest(BaseModel):
    sentences: list[str]


POSITIVE_WORDS = {
    "love", "loved", "lovely", "great", "excellent", "amazing",
    "awesome", "good", "happy", "wonderful", "fantastic", "best",
    "beautiful", "perfect", "enjoy", "enjoyed", "like", "liked",
    "excited", "delighted", "pleased", "brilliant", "outstanding",
    "success", "successful", "win", "winning", "glad", "grateful",
    "thankful", "helpful", "recommend", "impressive", "easy", "fun",
    "positive", "kind", "friendly", "superb", "hopeful", "favorite",
    "favourite", "nice", "sweet", "smile", "smiling", "celebrate"
}

NEGATIVE_WORDS = {
    "hate", "hated", "terrible", "awful", "bad", "horrible",
    "sad", "angry", "disappointed", "disappointing", "worst",
    "poor", "boring", "annoying", "annoyed", "fail", "failed",
    "failure", "wrong", "problem", "problems", "painful", "ugly",
    "useless", "difficult", "unhappy", "upset", "fear", "afraid",
    "scared", "frustrated", "frustrating", "broken", "dislike",
    "disliked", "regret", "regretful", "sadly", "unfortunately",
    "negative", "disgusting", "pathetic", "sick", "worse", "stress",
    "stressed", "stressful", "cry", "crying", "fail", "failure",
    "lonely", "miserable", "disaster", "annoy", "pain", "hurt"
}

NEGATIONS = {
    "not", "never", "no", "isn't", "wasn't", "don't",
    "doesn't", "didn't", "can't", "couldn't", "won't",
    "wouldn't", "hardly", "neither", "nor"
}

INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "so",
    "absolutely", "truly", "highly", "super"
}


def analyze_sentiment(sentence: str) -> str:
    text = sentence.lower()
    words = re.findall(r"\b[\w']+\b", text)

    positive_score = 0.0
    negative_score = 0.0

    for i, word in enumerate(words):
        if word not in POSITIVE_WORDS and word not in NEGATIVE_WORDS:
            continue

        score = 1.5 if i > 0 and words[i - 1] in INTENSIFIERS else 1.0
        preceding = words[max(0, i - 3):i]
        negated = any(w in NEGATIONS for w in preceding)

        if word in POSITIVE_WORDS:
            if negated:
                negative_score += score
            else:
                positive_score += score
        else:
            if negated:
                positive_score += score
            else:
                negative_score += score

    # Handle a few common expressions.
    if "thank you" in text or "well done" in text:
        positive_score += 1

    if "looking forward" in text or "can't wait" in text:
        positive_score += 1

    if positive_score > negative_score:
        return "happy"
    if negative_score > positive_score:
        return "sad"
    return "neutral"


@app.post("/sentiment")
def sentiment_analysis(request: SentimentRequest):
    return {
        "results": [
            {
                "sentence": sentence,
                "sentiment": analyze_sentiment(sentence),
            }
            for sentence in request.sentences
        ]
    }

