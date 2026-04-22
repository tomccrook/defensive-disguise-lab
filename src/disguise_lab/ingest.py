"""Hash-verified retrieval of the specifically authorised historical CSV copy."""

import concurrent.futures
import json
import time
from pathlib import Path

import httpx

from disguise_lab.audit import file_hash


def retrieve_release(root: Path, evidence: Path, workers: int = 2) -> dict:
    """Anonymous public HTTPS; fixed source revision; no credentials or rule actions."""
    if workers not in (1, 2):
        raise ValueError("retrieval permits at most two concurrent files")
    source = json.loads(evidence.read_text())
    revision = source["source"]["revision"]
    root.mkdir(parents=True, exist_ok=True)
    base = (
        "https://media.githubusercontent.com/media/b00nk3r/"
        f"NFL-Big-Data-Bowl-2025/{revision}/code/data/"
    )

    def fetch(record: dict) -> dict:
        name = record["file"]
        if Path(name).name != name or not name.endswith(".csv"):
            raise ValueError("source manifest has an invalid filename")
        expected = record.get("expected_sha256", record.get("http_etag", "").strip('"'))
        if len(expected) != 64:
            raise ValueError("full source SHA256 is required")
        path = root / name
        receipt = root / f"{name}.receipt.json"
        start = time.perf_counter()
        if path.exists():
            if path.stat().st_size != record["advertised_bytes"] or file_hash(path) != expected:
                raise ValueError(f"existing source checksum mismatch: {name}")
            return json.loads(receipt.read_text())
        partial = root / f"{name}.partial"
        try:
            import hashlib

            checksum = hashlib.sha256()
            size = 0
            with httpx.Client(timeout=120, follow_redirects=True) as client:
                with client.stream("GET", base + name) as response:
                    response.raise_for_status()
                    with partial.open("wb") as stream:
                        for chunk in response.iter_bytes(1024 * 1024):
                            size += len(chunk)
                            if size > record["advertised_bytes"]:
                                raise ValueError(f"source exceeds declared size: {name}")
                            stream.write(chunk)
                            checksum.update(chunk)
            actual = checksum.hexdigest()
            if size != record["advertised_bytes"] or actual != expected:
                raise ValueError(f"source checksum/size mismatch: {name}")
            partial.rename(path)
            result = {
                "file": name,
                "bytes": size,
                "sha256": actual,
                "status": "verified",
                "source_revision": revision,
                "seconds": time.perf_counter() - start,
            }
            receipt.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            return result
        except BaseException as error:
            failure = {
                "file": name,
                "status": "failed",
                "error_type": type(error).__name__,
                "seconds": time.perf_counter() - start,
            }
            # Do not log redirect URLs or response bodies.
            (root / f"{name}.failure-{time.time_ns()}.json").write_text(json.dumps(failure))
            raise

    metadata = [r for r in source["files"] if not r["file"].startswith("tracking_")]
    tracking = [r for r in source["files"] if r["file"].startswith("tracking_")]
    results = [fetch(r) for r in metadata]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results.extend(pool.map(fetch, tracking))
    manifest = {
        "source_revision": revision,
        "files": results,
        "status": "verified",
        "rights_mode": "local_noncommercial_research_only; no_public_raw_replay",
    }
    (root / "retrieval.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
