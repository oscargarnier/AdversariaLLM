from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from adversariallm.attacks.attack import AttackStepResult
from adversariallm.io_utils.config import RunConfig
from adversariallm.io_utils import config as config_utils


def test_attack_step_result_backwards_compatible_old_payload():
    old_step = {"step": 0, "model_completions": ["ok"]}
    step = AttackStepResult(**old_step)
    assert step.model_completions == ["ok"]
    assert step.model_completions_raw is None
    assert step.defense_metadata is None


def test_run_config_without_schema_version_fields():
    rc = RunConfig(
        model="m",
        dataset="d",
        attack="a",
        defense=None,
        model_params={},
        dataset_params={},
        attack_params={},
    )
    assert "config_schema_version" not in rc.__dict__
    assert "defense_schema_version" not in rc.__dict__


def test_run_config_separates_attack_and_runtime_defense_configs(monkeypatch):
    run_config = RunConfig(
        model="m",
        dataset="d",
        attack="a",
        defense=None,
        model_params={"id": "m"},
        dataset_params=OmegaConf.create({"idx": [0]}),
        attack_params={"name": "a"},
        experiment_type="runtime_inference",
        runtime_defense="polyguard",
        runtime_defense_params={"name": "polyguard"},
    )

    attack_config = run_config.to_attack_config()
    runtime_config = run_config.to_mongo_config()

    assert attack_config["experiment_type"] == "attack_compute"
    assert "runtime_defense" not in attack_config
    assert "runtime_defense_params" not in attack_config
    assert runtime_config["experiment_type"] == "runtime_inference"
    assert runtime_config["runtime_defense"] == "polyguard"
    assert runtime_config["runtime_defense_params"]["name"] == "polyguard"

    class _FakeCollection:
        def __init__(self):
            self.docs = [
                {"config": attack_config, "log_file": "/tmp/attack.json"},
                {"config": runtime_config, "log_file": "/tmp/runtime.json"},
            ]

        def find_one(self, query):
            for doc in self.docs:
                if doc["config"] == query["config"]:
                    return doc
            return None

    monkeypatch.setattr(config_utils, "get_mongodb_connection", lambda: SimpleNamespace(runs=_FakeCollection()))

    filtered = config_utils.filter_config_for_runtime(run_config, 1, overwrite=False)
    assert filtered is not None
    assert filtered.runtime_defense == "polyguard"


def test_filter_config_rejects_runtime_defense_fields(monkeypatch):
    run_config = RunConfig(
        model="m",
        dataset="d",
        attack="a",
        defense=None,
        model_params={"id": "m"},
        dataset_params=OmegaConf.create({"idx": [0]}),
        attack_params={"name": "a"},
        experiment_type="runtime_inference",
        runtime_defense="polyguard",
        runtime_defense_params={"name": "polyguard"},
    )

    monkeypatch.setattr(config_utils, "get_mongodb_connection", lambda: SimpleNamespace(runs=SimpleNamespace(find_one=lambda _q: None)))

    with pytest.raises(ValueError, match="runtime defense fields must be absent"):
        config_utils.filter_config(run_config, 1, overwrite=False)
