from __future__ import annotations

from omegaconf import OmegaConf

import run_inference


def test_collect_configs_cross_products_attack_and_runtime_defenses(monkeypatch):
    class _DummyDataset:
        config_idx = [0]

        def __len__(self):
            return 1

    class _DummyPromptDataset:
        @staticmethod
        def from_name(_name):
            return lambda _cfg: _DummyDataset()

    monkeypatch.setattr(run_inference, "PromptDataset", _DummyPromptDataset)
    monkeypatch.setattr(run_inference, "filter_config_for_runtime", lambda rc, _dset_len, overwrite=False: rc)

    cfg = OmegaConf.create(
        {
            "models": {"m": {"id": "m"}},
            "datasets": {"d": {"idx": [0]}},
            "attacks": {"a": {"name": "a"}},
            "defenses": {"none": {"name": "none"}, "attack_polyguard": {"name": "polyguard"}},
            "runtime_defenses": {"none": {"name": "none"}, "runtime_polyguard": {"name": "polyguard"}},
            "model": "m",
            "dataset": "d",
            "attack": "a",
            "defense": None,
            "runtime_defense": None,
            "experiment_type": "runtime_inference",
            "overwrite": True,
        }
    )

    run_configs = run_inference.collect_configs(cfg)

    assert len(run_configs) == 4
    assert {rc.experiment_type for rc in run_configs} == {"runtime_inference"}
    assert {rc.defense for rc in run_configs} == {None, "polyguard"}
    assert {rc.runtime_defense for rc in run_configs} == {None, "polyguard"}