# recurrentlens

**Mechanistic interpretability for State-Space Models.** Sparse autoencoders, feature visualization, and a Hub registry for Mamba / Mamba-2.

[![CI](https://github.com/hinanohart/recurrentlens/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/recurrentlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

**Colab notebooks:** [Quickstart](https://colab.research.google.com/github/hinanohart/recurrentlens/blob/main/notebooks/01_quickstart.ipynb) · [Explore an SAE](https://colab.research.google.com/github/hinanohart/recurrentlens/blob/main/notebooks/02_explore_pretrained.ipynb) · [Train your own](https://colab.research.google.com/github/hinanohart/recurrentlens/blob/main/notebooks/03_train_mamba130m_sae.ipynb)

---

## Status

**Alpha.** The public API is intentionally small (see [v0.1.0 API](#v010-api)) and may evolve through v0.1.x. Pretrained SAE artifacts are **not** yet on the Hub — they ship in v0.1.1 via a community Colab T4 run. See [CHANGELOG.md](./CHANGELOG.md) for the v0.1.0.post1 correctness hotfix (JumpReLU STE, HTML escape, CE metric cleanup).

## ⚠️ v0.1.0.post1 scope disclosure (read first)

`recurrentlens` ships a **framework + smoke-tested API + 3 Colab notebooks**. **Pretrained SAE artifacts for Mamba-130M/1.3B will land in v0.1.1**, trained via the included Colab notebook (T4 GPU is sufficient). The maintainer's build environment is CPU-only; the artifact training is deferred rather than fabricated.

If you have a GPU and want to seed the Hub registry early, run `notebooks/03_train_mamba130m_sae.ipynb` and `recurrentlens.hub.push_sae(...)` — pull requests welcome.

## Why recurrentlens

SAELens is the canonical SAE training library and supports a wide range of architectures. As of 2026-05, the Mamba path has [open issues since 2024-10](https://github.com/decoderesearch/SAELens/issues/311), and there is room for an SSM-first design that treats recurrent state and selective scan as first-class concerns rather than retrofitting them onto a transformer-shaped abstraction. `recurrentlens` is that design.

## Install

```bash
uv pip install recurrentlens                # core (CPU smoke + scaffold)
uv pip install "recurrentlens[mamba]"        # adds mamba-ssm CUDA kernels — Linux + CUDA only
uv pip install "recurrentlens[dev]"          # development
```

> The `[mamba]` extra pulls `mamba-ssm`, which requires CUDA and a Linux toolchain. CPU users and macOS users should install just `recurrentlens` (the core works via HF Transformers' Mamba implementation).

## Quick look

```python
import recurrentlens as rl

model = rl.load_model("state-spaces/mamba-130m-hf")
sae = rl.train_sae(
    model,
    hook_site="out_proj_out",   # residual-stream analog (default)
    layer=6,
    variant="topk",
    d_sae=16384,
    n_tokens=200_000,            # CPU smoke; use 5M-100M+ on a T4 GPU
)

with rl.steer(model, sae, feature_id=42, vector_scale=2.0):
    out = model.generate("The capital of France is", max_new_tokens=20)

# Capture activations + contexts first; see notebooks/02 for a runnable example.
# rl.viz.feature_explorer(sae, feature_id=42, activations=acts, contexts=ctxs).save("feature42.html")
```

> **`n_tokens` note.** 5M tokens on CPU takes hours; 200k completes in a couple of minutes for a smoke check. For research-grade SAEs the design target is 100M+ tokens, which is what the v0.1.1 pretrained artifacts will be trained on via Colab T4. See `notebooks/03_train_mamba130m_sae.ipynb`.

See `notebooks/` for end-to-end examples.

## v0.1.0 API

| Function | Status |
|---|---|
| `load_model(model_id, device, dtype)` | scaffold (Phase 2) |
| `train_sae(model, hook_site, layer, variant, ...)` | scaffold (Phases 4–5) |
| `ablate(model, sae, feature_id, strength)` | scaffold (Phase 7) |
| `steer(model, sae, feature_id, vector_scale)` | scaffold (Phase 7) |
| `viz.feature_explorer(sae, feature_id, ...)` | scaffold (Phase 8) |
| `hub.load_sae(repo_id)` / `hub.push_sae(sae, repo_id)` | scaffold (Phase 8) |
| `bench.evaluate(sae, metrics=[...])` | scaffold (Phase 9) |

`extract_circuit` and `diff.compare` are scheduled for **v0.2**.

## Supported models (v0.1.0)

- `state-spaces/mamba-130m-hf` (smoke + Colab training)
- `state-spaces/mamba-130m` (raw)
- `state-spaces/mamba2-1.3b` (Colab training only — too large for CI smoke)

Falcon-Mamba-7B, RWKV-7, and Jamba are **v0.2+** targets.

## Honest assessment

- **Novelty**: As of 2026-05, OSS coverage of SSM-specific SAE training is thin; SAELens has Mamba code paths but [Issue #311](https://github.com/decoderesearch/SAELens/issues/311) has been open since 2024-10. `recurrentlens` treats the SSM design (recurrent state, selective scan, `out_proj` residual) as first-class.
- **Practicality**: `load_model` + `train_sae` + `viz.feature_explorer` + `hub.{load,push}_sae` gives users a runnable scaffold today; with v0.1.1 pretrained artifacts the path becomes a 5-minute end-to-end demo.
- **Scope**: v0.1.0.post1 has **no** pretrained SAE artifacts. Real recon-MSE / L0 / CE-recovery numbers will be published with v0.1.1 SAEs (trained via Colab T4). The shipped `train_sae` default is `n_tokens=200_000` for CPU smoke; the research-grade target is 100M+ tokens, which v0.1.1 will use.
- **Open question**: do Mamba `out_proj_out` activations actually yield monosemantic SAE features? An empirical question the v0.1.1 release will start answering.

## Pretrained SAEs

| repo_id | model | layer | hook | variant | d_sae | status |
|---|---|---|---|---|---|---|
| `hinanohart/recurrentlens-mamba130m-L2-sae` | mamba-130m | 2 | out_proj_out | topk | 16384 | **planned v0.1.1** |
| `hinanohart/recurrentlens-mamba130m-L6-sae` | mamba-130m | 6 | out_proj_out | topk | 16384 | **planned v0.1.1** |
| `hinanohart/recurrentlens-mamba130m-L10-sae` | mamba-130m | 10 | out_proj_out | topk | 16384 | **planned v0.1.1** |

Want to seed this table? Run notebook 03 on Colab T4 and open a PR. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

See `CITATION.cff`.

## License

Apache-2.0.
