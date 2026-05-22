"""Phase 8 viz + Phase 9 bench unit tests."""

import torch

from recurrentlens.bench import BenchReport, evaluate
from recurrentlens.sae.variants import VanillaSAE
from recurrentlens.viz import HTMLOutput, feature_explorer


def test_feature_explorer_header_only():
    sae = VanillaSAE(d_in=8, d_sae=16, hook_site="out_proj_out", layer=2, model_id="x/y")
    out = feature_explorer(sae, feature_id=3)
    assert isinstance(out, HTMLOutput)
    assert "Feature 3" in out.html
    assert "x/y" in out.html
    assert "out_proj_out" in out.html
    assert "No activations provided" in out.html


def test_feature_explorer_with_activations(tmp_path):
    torch.manual_seed(0)
    sae = VanillaSAE(d_in=8, d_sae=16)
    acts = torch.randn(50, 8)
    ctx = [f"context {i}" for i in range(50)]
    out = feature_explorer(sae, feature_id=2, activations=acts, contexts=ctx, top_k=5)
    assert "context" in out.html
    # save roundtrip
    p = tmp_path / "f.html"
    out.save(str(p))
    assert p.read_text(encoding="utf-8") == out.html


def test_evaluate_returns_report_with_recon_and_l0():
    sae = VanillaSAE(d_in=8, d_sae=16)
    acts = torch.randn(32, 8)
    report = evaluate(sae, activations=acts)
    assert isinstance(report, BenchReport)
    assert "reconstruction_mse" in report.metrics
    assert "l0" in report.metrics
    assert "fvu" in report.metrics
    assert "ce_recovery skipped" in " ".join(report.notes)


def test_evaluate_subset_metrics():
    sae = VanillaSAE(d_in=8, d_sae=16)
    acts = torch.randn(32, 8)
    report = evaluate(sae, activations=acts, metrics=["l0"])
    assert "l0" in report.metrics
    assert "reconstruction_mse" not in report.metrics


def test_bench_report_dunder():
    r = BenchReport(metrics={"a": 0.5, "b": 0.25})
    assert r["a"] == 0.5
    assert r.to_dict() == {"metrics": {"a": 0.5, "b": 0.25}, "notes": []}
    assert "a=0.5" in repr(r)
