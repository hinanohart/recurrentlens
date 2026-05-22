"""Unit tests for the model wrapper (no HF download)."""

from types import SimpleNamespace

import pytest

from recurrentlens.core.model import RecurrentLensModelImpl, _MambaConfigShim


def _fake_hf_config(n_layers=4, d_model=128, vocab_size=50000):
    return SimpleNamespace(
        num_hidden_layers=n_layers,
        hidden_size=d_model,
        vocab_size=vocab_size,
        model_type="mamba",
    )


def _fake_legacy_config(n_layers=4, d_model=128, vocab_size=50000):
    return SimpleNamespace(
        n_layer=n_layers,
        d_model=d_model,
        vocab_size=vocab_size,
    )


def test_config_shim_hf_layout():
    cfg = _MambaConfigShim.from_hf_config(_fake_hf_config())
    assert cfg.n_layers == 4
    assert cfg.d_model == 128
    assert cfg.vocab_size == 50000


def test_config_shim_legacy_layout():
    cfg = _MambaConfigShim.from_hf_config(_fake_legacy_config(n_layers=6, d_model=256))
    assert cfg.n_layers == 6
    assert cfg.d_model == 256


def test_wrapper_attrs():
    cfg = _MambaConfigShim(n_layers=3, d_model=64, vocab_size=100)
    fake_layers = [SimpleNamespace(name=f"layer{i}") for i in range(3)]
    fake_hf = SimpleNamespace(backbone=SimpleNamespace(layers=fake_layers), config=cfg)

    m = RecurrentLensModelImpl(
        hf_model=fake_hf,
        tokenizer=None,
        model_id="fake/mamba",
        device="cpu",
        dtype="float32",
        config=cfg,
    )
    assert m.n_layers == 3
    assert m.d_model == 64
    assert m.hook_sites() == ["out_proj_out", "ssm_h_t"]
    assert "fake/mamba" in repr(m)


def test_get_layer_backbone_layout():
    cfg = _MambaConfigShim(n_layers=2, d_model=8, vocab_size=10)
    fake_layers = [SimpleNamespace(idx=0), SimpleNamespace(idx=1)]
    fake_hf = SimpleNamespace(backbone=SimpleNamespace(layers=fake_layers), config=cfg)
    m = RecurrentLensModelImpl(
        hf_model=fake_hf, tokenizer=None, model_id="x", device="cpu", dtype="f", config=cfg
    )
    assert m.get_layer(0).idx == 0
    assert m.get_layer(1).idx == 1


def test_get_layer_out_of_range():
    cfg = _MambaConfigShim(n_layers=2, d_model=8, vocab_size=10)
    fake_hf = SimpleNamespace(
        backbone=SimpleNamespace(layers=[SimpleNamespace(), SimpleNamespace()]),
        config=cfg,
    )
    m = RecurrentLensModelImpl(
        hf_model=fake_hf, tokenizer=None, model_id="x", device="cpu", dtype="f", config=cfg
    )
    with pytest.raises(IndexError):
        m.get_layer(5)
