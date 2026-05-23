# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project version follows [PEP 440](https://peps.python.org/pep-0440/)
(SemVer-compatible, with `post` releases reserved for correctness-only patches).

## [0.1.0.post1] — 2026-05-23

Correctness hotfix on top of v0.1.0. No new features, no API additions, no
breaking changes to the eight public functions. The release fixes silent
correctness bugs uncovered in a post-release multi-agent audit.

### Fixed
- **JumpReLU SAE — straight-through estimator** (`sae/variants.py`).
  The forward gate is now `z = z_pre * gate`, the canonical STE form. The
  v0.1.0 implementation multiplied the surrogate gradient by `0.0`, which
  silently severed the backward path. `JumpReLUSAE` is now trainable.
- **`bench.evaluate` — removed `ce_recovery_proxy`** (`bench/evaluate.py`).
  The v0.1.0 ratio `ce_clean / ce_sae` was not the standard CE-recovery
  formula and produced numbers that could not be compared against SAELens,
  Anthropic, or DeepMind reports. The metric has been replaced by
  `ce_delta = ce_sae − ce_clean` (lower is better, 0 is perfect). The full
  zero-ablation baseline ships in v0.1.1.
- **`viz.feature_explorer` — HTML escape**.
  All user-controlled fields (`context`, `model_id`, `hook_site`, `variant`)
  are now passed through `html.escape`. This closes an XSS vector for
  feature pages rendered from Hub-loaded SAEs.
- **`BaseSAE.load` — metadata validation** (`sae/base.py`).
  Refuse to construct an SAE when the safetensors metadata is inconsistent
  with the stored tensors (e.g., a lying `d_sae` field) or names an unknown
  variant. Closes a supply-chain footgun for `hub.load_sae`.
- **L0 metric — signed nonzero count** (`sae/base.py`, `bench/evaluate.py`).
  `(z != 0)` instead of `(z > 0)`. Matches the standard SAE convention so
  negative activations (JumpReLU / Vanilla) are counted correctly.
- **`ssm_h_t` hook — warn once per module** (`hooks/registry.py`).
  The proxy warning previously fired on every `_resolve_target` call (i.e.
  every forward pass for ablate/steer/eval). It now fires once per layer
  module.
- **`train._cosine_lr` off-by-one** (`sae/train.py`).
  The final training step no longer collapses to `LR=0`; the schedule floors
  at 1% of `base_lr` at progress=1.
- **`features.ablate` — `torch.no_grad()`** (`features/ablate.py`).
  SAE encode/decode inside the intervention hook no longer builds an
  autograd graph.
- **`ActivationCache(mmap)` — overwrite warning** (`sae/cache.py`).
  Truncating an existing `.memmap` file now emits a `UserWarning` instead
  of silently destroying data.
- **`hub.push_sae` — `tempfile.TemporaryDirectory`** (`hub/io.py`).
  No more stale `/tmp/<repo_id>/...` directories left behind after upload.

### Changed
- **`hooks.resolve_target`** is now a public symbol (`recurrentlens.hooks`).
  The previous `_resolve_target` private name is kept as a back-compat alias
  until v0.2.
- **README `Quick look`**: `n_tokens=200_000` (CPU smoke). The 100M-token
  research-grade target is documented as the v0.1.1 pretrained-artifact path.
- **README SAELens characterization**: factual link to
  [decoderesearch/SAELens#311](https://github.com/decoderesearch/SAELens/issues/311)
  with the open-since date; removed editorial "broken" wording.
- **README** now carries an `Alpha` status badge near the top.
- **`notebooks/02_explore_pretrained.ipynb`**: `REPO_ID = None` by default,
  falls back to an untrained `TopKSAE` for the API demo. Pretrained
  Hub artifacts ship in v0.1.1.

### Added
- `tests/test_sae_unit.py::test_jumprelu_gradient_flows_to_encoder`
- `tests/test_sae_unit.py::test_l0_counts_signed_nonzero`
- `tests/test_sae_unit.py::test_base_sae_load_rejects_inconsistent_meta`
- `tests/test_sae_unit.py::test_base_sae_load_rejects_unknown_variant`
- `tests/test_viz_bench.py::test_feature_explorer_escapes_xss`
- `tests/test_viz_bench.py::test_feature_explorer_no_acts_emits_warning_banner`
- `tests/test_viz_bench.py::test_evaluate_does_not_emit_ce_recovery_proxy`
- `tests/test_hooks_unit.py::test_ssm_h_t_warning_is_emitted_once_per_module`
- `tests/test_train_smoke.py::test_cosine_lr_final_step_above_zero`
- `tests/test_features.py::test_ablate_zero_removes_feature_contribution`
  upgraded from a shape-only tautology to an analytic-delta assertion.

### Deferred to v0.1.1
- Pretrained Mamba-130M SAEs at layers {2, 6, 10} via Colab T4.
- TopK auxiliary-K dead-feature loss (Gao et al. 2024 §3.2).
- True CE recovery formula with zero-ablation baseline.
- Tangent-projected decoder gradient + checkpoint save in `train_sae_full`.
- nnsight backend for exact `h_t` capture.

## [0.1.0] — 2026-05-23

Initial public release. SSM-first SAE + feature-viz + Hub registry framework
for Mamba and Mamba-2. Apache-2.0, 38 unit tests, 3 Colab notebooks.
