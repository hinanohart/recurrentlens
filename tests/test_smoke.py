"""Smoke tests: imports, version, public API surface."""

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


def test_viz_callable():
    fn = rl.viz.feature_explorer
    assert callable(fn)


def test_hub_callable():
    assert callable(rl.hub.load_sae)
    assert callable(rl.hub.push_sae)


def test_bench_callable():
    assert callable(rl.bench.evaluate)
