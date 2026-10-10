
import json
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

# Telemetry JSON file is stored in the same directory as index.py.
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

        latencies = [
            record["latency_ms"] for record in matching
        ]
        uptimes = [
            record["uptime_pct"] for record in matching
        ]

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
