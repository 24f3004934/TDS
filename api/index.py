
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


# Sentiment analysis API
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
    "favourite", "nice", "sweet", "smile", "smiling", "celebrate",
    "incredible", "exceptional", "cheerful", "peaceful", "pleasure",
    "appreciate", "appreciated", "thrilled", "thanks", "thank",
    "satisfied", "satisfying", "convenient", "reliable", "delightful",
    "enjoyable", "charming", "inspiring", "inspired", "inspiration",
    "cool", "clever", "smart", "creative", "calm", "comfortable",
    "relaxed", "relaxing", "peace", "winner", "rewarding", "wow",
    "yay", "fast", "affordable", "smooth", "smoothly", "recommendable"
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
    "stressed", "stressful", "cry", "crying", "lonely", "miserable",
    "disaster", "annoy", "pain", "hurt", "horribly", "terribly",
    "ridiculous", "waste", "wasted", "slow", "expensive", "dreadful",
    "disgusted", "irritated", "irritating", "unpleasant", "unfortunate",
    "disappointment", "crash", "crashed", "damaged", "damage",
    "complaint", "complaints", "weak", "inferior", "mess", "messy",
    "mad", "furious", "worried", "worry", "anxious", "anxiety",
    "nervous", "scam", "fraud", "rejected", "lose", "lost", "loss",
    "overwhelmed", "exhausted", "tired", "confusing", "confused",
    "confusion", "rude", "cruel", "evil", "unacceptable",
    "disastrous", "mediocre", "incompetent", "suffer", "suffering",
    "uncomfortable", "unreliable", "inadequate", "inefficient",
    "miss", "missing", "awfully", "annoyance", "inferiority",
    "boring", "stressful", "regretful", "worthless", "disrespectful",
    "disrespect", "hostile", "threatening", "threat", "dangerous",
    "danger", "scary", "terrifying", "terrified", "desperate",
    "hopeless", "hopelessness", "heartbroken", "devastated",
    "devastating", "tragic", "tragedy", "sorrow", "grief",
    "gloomy", "depressing", "depressed", "disturbing", "disturbed",
    "offensive", "offended", "discomfort", "complain", "complained"
}

NEGATIONS = {
    "not", "never", "no", "isn't", "wasn't", "don't",
    "doesn't", "didn't", "can't", "couldn't", "won't",
    "wouldn't", "hardly", "neither", "nor", "cannot"
}

INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "so",
    "absolutely", "truly", "highly", "super", "quite",
    "remarkably", "deeply", "especially", "totally"
}


def analyze_sentiment(sentence: str) -> str:
    text = sentence.lower()
    words = re.findall(r"\b[\w']+\b", text)

    positive_score = 0.0
    negative_score = 0.0

    positive_phrases = (
        "thank you",
        "well done",
        "looking forward",
        "can't wait",
        "highly recommend",
        "works perfectly",
        "love it",
        "good job",
        "made my day",
        "worth it",
        "pleased with",
        "happy with",
        "very helpful",
        "great experience",
        "well worth",
        "enjoyed it",
        "glad to hear",
        "good news",
        "feel great",
        "feel good",
        "very good",
        "very nice",
        "excellent service",
        "keep up the good work",
    )

    negative_phrases = (
        "not worth",
        "waste of time",
        "waste of money",
        "worst ever",
        "not good",
        "not great",
        "not happy",
        "not satisfied",
        "poor quality",
        "doesn't work",
        "does not work",
        "never again",
        "fed up",
        "not recommend",
        "feel terrible",
        "very disappointed",
        "not worth it",
        "disappointed with",
        "not impressed",
        "not helpful",
        "not working",
        "bad experience",
        "terrible experience",
        "very bad",
        "very poor",
        "not pleased",
        "waste of effort",
        "feel awful",
        "feel sad",
        "really bad",
        "not acceptable",
    )

    for phrase in positive_phrases:
        if phrase in text:
            positive_score += 2

    for phrase in negative_phrases:
        if phrase in text:
            negative_score += 2

    for i, word in enumerate(words):
        if word not in POSITIVE_WORDS and word not in NEGATIVE_WORDS:
            continue

        score = (
            1.5
            if i > 0 and words[i - 1] in INTENSIFIERS
            else 1.0
        )

        preceding = words[max(0, i - 3):i]
        negated = any(w in NEGATIONS for w in preceding)

        if word in POSITIVE_WORDS:
            if negated:
                negative_score += score
            else:
                positive_score += score
        elif word in NEGATIVE_WORDS:
            if negated:
                positive_score += score
            else:
                negative_score += score

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
