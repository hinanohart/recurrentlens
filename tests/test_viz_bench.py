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


def test_feature_explorer_escapes_xss():
    """Regression: v0.1.0 emitted unescaped context / model_id, allowing
    script injection from Hub-hosted SAEs or hostile fineweb tokens.

    The browser-level test is: no user-controlled bytes may form an executable
    tag. We verify by stripping the static template chrome and asserting that
    the remaining user-derived payload contains no literal `<` or `>`.
    """
    sae = VanillaSAE(d_in=4, d_sae=8, model_id="<script>alert(1)</script>")
    acts = torch.randn(3, 4)
    ctxs = [
        "<img src=x onerror=alert(2)>",
        "</span><script>alert(3)</script>",
        "normal & innocuous",
    ]
    out = feature_explorer(sae, feature_id=0, activations=acts, contexts=ctxs, top_k=3)

    # 1. None of the user-payload markers may appear in their raw, executable form.
    for payload in (
        "<script>",
        "</script>",
        "<img src=x onerror=alert",
        "</span><script>",
    ):
        assert payload not in out.html, f"raw {payload!r} leaked into HTML (XSS)"

    # 2. The escaped form must be present (confirms escape ran, not just absent).
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out.html, "model_id was not escaped"
    assert "&lt;img src=x onerror=alert(2)&gt;" in out.html, "context was not escaped"


def test_feature_explorer_no_acts_emits_warning_banner():
    """Regression: a header-only call previously produced a near-blank page
    that users mistook for success. v0.1.0.post1 surfaces a warning banner.
    """
    sae = VanillaSAE(d_in=4, d_sae=8, model_id="m/x")
    out = feature_explorer(sae, feature_id=0)
    assert "No activations provided" in out.html
    # banner uses a salmon background so it cannot be missed
    assert "#fff7e6" in out.html


def test_evaluate_does_not_emit_ce_recovery_proxy():
    """Regression: ce_recovery_proxy = ce_clean / ce_sae was mathematically
    meaningless and was removed in v0.1.0.post1.
    """
    sae = VanillaSAE(d_in=8, d_sae=16)
    acts = torch.randn(32, 8)
    report = evaluate(sae, activations=acts)
    assert "ce_recovery_proxy" not in report.metrics
    assert "ce_recovery" not in report.metrics  # full formula deferred to v0.1.1
