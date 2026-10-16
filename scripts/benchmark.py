"""Measure loopback HTTP inference on the active approved study; never a GPU/live latency claim."""

import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np

with httpx.Client(base_url="http://127.0.0.1:8008", timeout=30) as client:
    info = client.get("/api/data").raise_for_status().json()
    plays = client.get("/api/plays?view=league&held_out=true").raise_for_status().json()["plays"]
    samples = [
        client.get(f"/api/sample/{p['game_id']}/{p['play_id']}").raise_for_status().json()
        for p in plays[:21]
    ]
    records = []
    for family in info["models"]:
        first = client.post("/api/forecast", json={"family": family, "sample": samples[0]})
        first.raise_for_status()
        for condition in ["unique-input-warm-model", "cached-identical-input"]:
            timings = []
            values = samples[1:] if condition == "unique-input-warm-model" else [samples[0]] * 20
            for sample in values:
                start = time.perf_counter()
                result = client.post("/api/forecast", json={"family": family, "sample": sample})
                result.raise_for_status()
                timings.append((time.perf_counter() - start) * 1000)
            records.append(
                {
                    "family": family,
                    "condition": condition,
                    "requests": len(timings),
                    "median_ms": float(np.median(timings)),
                    "p95_ms": float(np.percentile(timings, 95)),
                    "model_id": first.json()["model_id"],
                }
            )
    output = {
        "scope": (
            "Sequential local loopback HTTP, includes JSON validation/serialization; "
            "excludes model load and sample retrieval; descriptive small local workload"
        ),
        "platform": platform.platform(),
        "dataset_id": info["dataset"]["dataset_id"],
        "device": "cpu",
        "records": records,
    }
    output_path = Path("docs/evidence") / (
        "http-inference-benchmark.json"
        if info["dataset"]["synthetic"]
        else "nfl-http-inference-benchmark.json"
    )
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Recorded local HTTP benchmark: {output_path}")
