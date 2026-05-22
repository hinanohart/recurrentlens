"""Smoke tests: imports, version, public API surface."""

import pytest

import recurrentlens as rl
from recurrentlens import SAE, RecurrentLensModel


def test_version():
    assert rl.__version__ == "0.1.0"


def test_protocols_importable():
    assert RecurrentLensModel is not None
    assert SAE is not None


def test_public_api_surface():
    """v0.1.0 public API is callable (raising NotImplementedError is OK at Phase 1)."""
    for name in ["load_model", "train_sae", "ablate", "steer"]:
        assert hasattr(rl, name), f"missing public API: {name}"
        assert callable(getattr(rl, name)), f"not callable: {name}"
    for submod in ["viz", "hub", "bench"]:
        assert hasattr(rl, submod), f"missing submodule: {submod}"


def test_load_model_phase1_stub():
    with pytest.raises(NotImplementedError, match="Phase 2"):
        rl.load_model("state-spaces/mamba-130m-hf")


def test_train_sae_phase1_stub():
    with pytest.raises(NotImplementedError, match="Phases 4-5"):
        rl.train_sae(model=None, hook_site="out_proj_out", layer=0)


def test_viz_deferred():
    with pytest.raises(NotImplementedError, match="Phase 8"):
        rl.viz.feature_explorer(None, 0)


def test_hub_deferred():
    with pytest.raises(NotImplementedError, match="Phase 8"):
        rl.hub.load_sae("some/repo")


def test_bench_deferred():
    with pytest.raises(NotImplementedError, match="Phase 9"):
        rl.bench.evaluate(None)
