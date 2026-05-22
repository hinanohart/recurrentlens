"""Hook registration unit tests (no real model)."""

import torch
import torch.nn as nn

from recurrentlens.hooks.registry import HookHandle, HookManager, register_hook


class _FakeMixer(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.out_proj = nn.Linear(d, d, bias=False)
        self.ssm = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.out_proj(self.ssm(x))


class _FakeBlock(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.mixer = _FakeMixer(d=d)

    def forward(self, x):
        return self.mixer(x)


class _FakeWrapper:
    """Minimal RecurrentLensModelImpl substitute exposing .get_layer()."""

    def __init__(self, n_layers=2, d=8):
        self.n_layers = n_layers
        self._layers = [_FakeBlock(d=d) for _ in range(n_layers)]

    def get_layer(self, idx):
        return self._layers[idx]


def test_register_hook_out_proj_out_captures_activations():
    model = _FakeWrapper(n_layers=2, d=8)
    handle = register_hook(model, site="out_proj_out", layer=0)
    assert isinstance(handle, HookHandle)
    x = torch.randn(2, 8)
    model.get_layer(0)(x)
    assert len(handle.activations) == 1
    assert handle.activations[0].shape == (2, 8)
    handle.remove()


def test_register_hook_ssm_h_t_emits_warning_and_captures():
    model = _FakeWrapper(n_layers=1, d=4)
    import warnings as w

    with w.catch_warnings(record=True) as records:
        w.simplefilter("always")
        handle = register_hook(model, site="ssm_h_t", layer=0)
    # at least one warning of UserWarning class
    assert any("proxy" in str(r.message) for r in records)
    x = torch.randn(3, 4)
    model.get_layer(0)(x)
    assert len(handle.activations) == 1
    handle.remove()


def test_hook_manager_context_removes_all():
    model = _FakeWrapper(n_layers=2, d=8)
    with HookManager(model) as mgr:
        h0 = mgr.add(site="out_proj_out", layer=0)
        h1 = mgr.add(site="out_proj_out", layer=1)
        model.get_layer(0)(torch.randn(1, 8))
        model.get_layer(1)(torch.randn(1, 8))
        assert len(h0.activations) == 1
        assert len(h1.activations) == 1
    # After exit, both handles removed (no more captures).
    h0.clear()
    h1.clear()
    model.get_layer(0)(torch.randn(1, 8))
    assert len(h0.activations) == 0


def test_transform_applied():
    model = _FakeWrapper(n_layers=1, d=8)
    handle = register_hook(
        model,
        site="out_proj_out",
        layer=0,
        transform=lambda t: t.detach() * 2,
    )
    x = torch.randn(1, 8)
    model.get_layer(0)(x)
    cap = handle.activations[0]
    # check that transform doubled the magnitude relative to a fresh forward
    fresh = model.get_layer(0)(x).detach()
    assert torch.allclose(cap, fresh * 2, atol=1e-5)
    handle.remove()
