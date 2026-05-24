import polars as pl
import pytest

from disguise_lab.audit import audit_directory, unique_keys
from disguise_lab.registry import Registry


def test_registry_budget_and_failed_history(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite", max_trials=1)
    identity = registry.start({"family": "graph"})
    registry.finish(identity, "failed", {"reason": "fixture failure"})
    assert registry.records()[0]["status"] == "failed"
    with pytest.raises(ValueError):
        registry.finish(identity, "succeeded", {})
    with pytest.raises(RuntimeError):
        registry.start({"family": "pooled"})


def test_duplicate_join_keys_and_missing_source(tmp_path):
    with pytest.raises(ValueError):
        unique_keys(pl.DataFrame({"gameId": [1, 1]}), ["gameId"], "games")
    assert audit_directory(tmp_path)["status"] == "missing_files"
