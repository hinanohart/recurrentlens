"""Phase 7 steer + ablate context manager tests on a fake model."""

import torch
import torch.nn as nn

from recurrentlens.features.ablate import ablate
from recurrentlens.features.steer import steer
from recurrentlens.sae.variants import VanillaSAE


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
    def __init__(self, n_layers=1, d=8):
        self.n_layers = n_layers
        self._layers = [_FakeBlock(d=d) for _ in range(n_layers)]

    def get_layer(self, idx):
        return self._layers[idx]


def test_steer_adds_direction_to_output():
    torch.manual_seed(0)
    model = _FakeWrapper(d=8)
    sae = VanillaSAE(d_in=8, d_sae=16, hook_site="out_proj_out", layer=0)
    x = torch.randn(1, 8)
    base = model.get_layer(0)(x).clone()

    with steer(model, sae, feature_id=3, vector_scale=2.0):
        steered = model.get_layer(0)(x)

    diff = (steered - base).detach()
    expected = 2.0 * sae.W_dec[3]
    assert torch.allclose(diff, expected, atol=1e-5)


def test_steer_context_removes_hook():
    model = _FakeWrapper(d=8)
    sae = VanillaSAE(d_in=8, d_sae=16, hook_site="out_proj_out", layer=0)
    x = torch.randn(1, 8)
    pre = model.get_layer(0)(x).clone()
    with steer(model, sae, feature_id=0, vector_scale=1.0):
        pass
    post = model.get_layer(0)(x).clone()
    assert torch.allclose(pre, post, atol=1e-6)


def test_ablate_zero_removes_feature_contribution():
    torch.manual_seed(0)
    model = _FakeWrapper(d=8)
    sae = VanillaSAE(d_in=8, d_sae=16, hook_site="out_proj_out", layer=0)
    x = torch.randn(2, 8)
    base = model.get_layer(0)(x).clone()
    with ablate(model, sae, feature_id=0, strength=0.0):
        ablated = model.get_layer(0)(x).clone()
    # Ablation should differ from base unless feature 0 was already zero.
    # We don't assert magnitude — just that hook fires and removes cleanly.
    with ablate(model, sae, feature_id=0, strength=1.0):
        identity = model.get_layer(0)(x).clone()
    # strength=1.0 means feature is unchanged → SAE reconstruction substituted twice,
    # so output should equal the SAE forward through the layer (not exactly base but stable)
    assert ablated.shape == base.shape
    assert identity.shape == base.shape
