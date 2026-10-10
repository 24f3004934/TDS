
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

DATA_FILE = Path(__file__).with_name("data.json")


class MetricsRequest(BaseModel):
    regions: list[str]
    threshold_ms: float = Field(ge=0)


def load_telemetry():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.post("/")
def calculate_metrics(request: MetricsRequest):
    try:
        records = load_telemetry()
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="Unable to load telemetry data",
        )

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

    return results
