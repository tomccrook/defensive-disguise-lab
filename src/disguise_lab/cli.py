"""Local reproducible jobs. Credentials and restricted data never enter CLI output."""

import json
import platform
from pathlib import Path

import torch
import torch_geometric
import typer

from disguise_lab.audit import audit_directory
from disguise_lab.models import MovementModel, masked_loss
from disguise_lab.study import run_synthetic, write_json

app = typer.Typer(no_args_is_help=True)


@app.command()
def nfl_retrieve(
    output: Path = Path("data/raw/nfl-big-data-bowl-2025"),
    source: Path = Path("docs/evidence/data-source-discovery.json"),
):
    """Retrieve the authorised fixed-revision copy; verify every full hash and byte count."""
    from disguise_lab.ingest import retrieve_release

    result = retrieve_release(output, source)
    typer.echo(f"{result['status']}: {len(result['files'])} files")


@app.command()
def nfl_canonical(
    raw: Path = Path("data/raw/nfl-big-data-bowl-2025"),
    output: Path = Path("data/canonical/nfl2022-v1"),
    config: Path = Path("configs/empirical.json"),
):
    """Build causal canonical partitions/cache and explicit source/cohort/annotation audits."""
    from disguise_lab.nfl import build_dataset

    typer.echo(str(build_dataset(raw, output, json.loads(config.read_text()))))


@app.command()
def nfl_train(
    data: Path = Path("data/canonical/nfl2022-v1"),
    output: Path = Path("artifacts/nfl2022-v1"),
    config: Path = Path("configs/empirical.json"),
):
    """Fit frozen game splits on CPU; preserve trials; never score final tests."""
    from disguise_lab.empirical import CachedNFL, fit_protocol, freeze_splits
    from disguise_lab.registry import Registry
    from disguise_lab.robustness import PLAN

    settings = json.loads(config.read_text())
    if (output / "test-exposure.json").exists():
        raise typer.BadParameter("final tests exposed; create a new prospectively defined study")
    torch.set_num_threads(settings["budget"]["cpu_threads"])
    cached = CachedNFL(data)
    splits = freeze_splits(cached, settings, output)
    plan = output / "analysis-plan.json"
    if plan.exists() and json.loads(plan.read_text()) != PLAN:
        raise typer.BadParameter("secondary plan differs from registered study")
    write_json(plan, PLAN)
    registry = Registry(
        Path("artifacts/trials.sqlite"), settings["budget"]["total_registry_trials"]
    )
    for protocol, split in splits.items():
        fit_protocol(cached, split, protocol, settings, output, registry, "cpu")
    write_json(
        output / "training-complete.json",
        {
            "protocols": list(splits),
            "trial_records": len(registry.records()),
            "test_exposed": False,
        },
    )
    typer.echo(f"Training complete; tests remain sealed: {output}")


@app.command()
def nfl_evaluate(
    data: Path = Path("data/canonical/nfl2022-v1"),
    run: Path = Path("artifacts/nfl2022-v1"),
    config: Path = Path("configs/empirical.json"),
):
    """Freeze models, record first test exposure, run proper scores and fixed sensitivities."""
    from disguise_lab.evaluation import evaluate

    torch.set_num_threads(4)
    try:
        result = evaluate(data, run, json.loads(config.read_text()))
    except Exception as error:
        import time
        import traceback

        write_json(
            run / f"evaluation-failure-{time.time_ns()}.json",
            {
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "test_exposed": (run / "test-exposure.json").exists(),
                "policy": "preserve partial artifacts and exposure; no automatic test reset",
            },
        )
        raise
    typer.echo(f"Locked evaluation {result['run_id']}: test exposure {result['test_exposure']}")


@app.command()
def nfl_corrective(
    data: Path = Path("data/canonical/nfl2022-v1"),
    run: Path = Path("artifacts/nfl2022-v1"),
):
    """Preserved post-exposure anchor-head control; never replaces original locked evidence."""
    from disguise_lab.corrective import run as correction

    correction(data, run)
    typer.echo("Corrective baseline audit saved separately; original model freeze is unchanged.")


@app.command()
def nfl_export(
    run: Path = Path("artifacts/nfl2022-v1"),
    output: Path = Path("docs/evidence"),
):
    """Export aggregate metrics, evidence manifests and scientific figures; no raw replay."""
    from disguise_lab.empirical_reporting import export_empirical

    export_empirical(run, output)
    typer.echo(f"Aggregate historical evidence exported: {output}")


@app.command()
def audit(path: Path, output: Path = Path("artifacts/audit.json")):
    """Audit locally downloaded source files; does not approve a research cohort."""
    result = audit_directory(path)
    write_json(output, result)
    typer.echo(f"{result['status']}: {output}")


@app.command()
def synthetic(output: Path = Path("artifacts"), epochs: int = 20, games: int = 18, plays: int = 8):
    """Run bounded synthetic models/calibration/evaluation, preserving all trials."""
    path = run_synthetic(output, epochs, games, plays)
    typer.echo(f"Synthetic study saved: {path}")


@app.command()
def host(output: Path = Path("artifacts/host.json"), device: str = "cpu"):
    """Verify a complete graph/GRU/probability/optimizer workload on requested device."""
    from torch_geometric.data import Batch

    from disguise_lab.features import build_features
    from disguise_lab.fixtures import generate, input_from_rows

    if device not in ["cpu", "cuda", "mps"]:
        raise typer.BadParameter("device must be cpu, cuda or mps")
    if device == "cuda" and not torch.cuda.is_available():
        raise typer.BadParameter("CUDA unavailable; do not train GPU workloads")
    if device == "mps" and not torch.backends.mps.is_available():
        raise typer.BadParameter("MPS unavailable")
    import time

    import polars as pl

    torch.manual_seed(1729)
    torch.set_num_threads(4)
    rows, meta = generate(6, 1)
    sample = input_from_rows(
        rows.filter(pl.col("game_id") == meta[0]["game_id"]), meta[0], -0.1, 1.0
    )
    batch = Batch.from_data_list([build_features(sample).graph()] * 8).to(device)
    model = MovementModel("graph").to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    start = time.perf_counter()
    mean, cov = model(batch)
    target = torch.zeros_like(mean)
    loss = masked_loss(mean, cov, target, torch.ones(len(mean), dtype=torch.bool, device=device))
    loss.backward()
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    result = {
        "status": "workload_passed",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pyg": torch_geometric.__version__,
        "device": device,
        "dtype": "float32",
        "seed": 1729,
        "batch_plays": 8,
        "nodes": batch.num_nodes,
        "edges": batch.num_edges,
        "seconds": time.perf_counter() - start,
        "loss": float(loss.detach().cpu()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
    }
    write_json(output, result)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def export(run: Path, output: Path = Path("docs/evidence")):
    """Export only synthetic metrics/manifests and publication-style figures."""
    from disguise_lab.reporting import export as export_report

    export_report(run, output)
    typer.echo(f"Synthetic evidence exported: {output}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8008):
    """Local read-only inference API; no external exposure by default."""
    import uvicorn

    uvicorn.run("disguise_lab.api:app", host=host, port=port)
