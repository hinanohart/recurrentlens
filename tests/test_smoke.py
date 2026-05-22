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
    for name in ["load_model", "train_sae", "ablate", "steer"]:
        assert hasattr(rl, name), f"missing public API: {name}"
        assert callable(getattr(rl, name)), f"not callable: {name}"
    for submod in ["viz", "hub", "bench"]:
        assert hasattr(rl, submod), f"missing submodule: {submod}"


def test_constants_exported():
    assert "out_proj_out" in rl.HOOK_SITES
    assert "ssm_h_t" in rl.HOOK_SITES
    assert "topk" in rl.SAE_VARIANTS
    assert rl.DEFAULT_HOOK_SITE == "out_proj_out"
    assert rl.DEFAULT_SAE_VARIANT == "topk"


def test_train_sae_phase5_stub():
    with pytest.raises(NotImplementedError, match="Phase 5"):
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


def test_ablate_deferred():
    with pytest.raises(NotImplementedError, match="Phase 7"):
        rl.ablate(None, None, 0)


def test_steer_deferred():
    with pytest.raises(NotImplementedError, match="Phase 7"):
        rl.steer(None, None, 0)
