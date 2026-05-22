# recurrentlens

**Mechanistic interpretability for State-Space Models.** Sparse autoencoders, feature visualization, and a Hub registry for Mamba / Mamba-2.

[![CI](https://github.com/hinanohart/recurrentlens/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/recurrentlens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## ⚠️ v0.1.0 scope disclosure (read first)

`recurrentlens` v0.1.0 ships a **framework + smoke-tested API + 3 Colab notebooks**. **Pretrained SAE artifacts for Mamba-130M/1.3B will land in v0.1.1**, trained via the included Colab notebook (T4 GPU is sufficient). The maintainer's build environment does not have a local GPU; the artifact training is deferred rather than fabricated.

If you have a GPU and want to seed the Hub registry early, run `notebooks/03_train_mamba130m_sae.ipynb` and `recurrentlens.hub.push_sae(...)` — pull requests welcome.

## Why recurrentlens

SAELens is the canonical SAE training library, but its Mamba integration is broken — see [decoderesearch/SAELens#311 (open since 2024-10)](https://github.com/decoderesearch/SAELens/issues/311). `recurrentlens` is **SSM-first**: hook sites, recurrent-state handling, and SAE training are designed for state-space models rather than retrofitted onto a transformer-shaped abstraction.

## Install

```bash
uv pip install recurrentlens                # core (CPU smoke + scaffold)
uv pip install "recurrentlens[mamba]"        # adds mamba-ssm CUDA kernels
uv pip install "recurrentlens[dev]"          # development
```

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
    n_tokens=5_000_000,
)

with rl.steer(model, sae, feature_id=42, vector_scale=2.0):
    out = model.generate("The capital of France is", max_new_tokens=20)

rl.viz.feature_explorer(sae, feature_id=42).save("feature42.html")
```

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

- **Novelty**: SSM-first SAE framework is OSS-blank as of 2026-05; SAELens has Mamba support but it is broken (Issue #311).
- **Practicality**: load + smoke + viz + Hub stub gives users a 5-minute path to a working SAE pipeline.
- **Limitation**: v0.1.0 has no pretrained SAE artifacts. Real recon-MSE / L0 / CE-recovery numbers will be published with v0.1.1 SAEs (trained via Colab T4).
- **Open question**: do Mamba `out_proj_out` activations actually yield monosemantic SAE features? This is an empirical question the v0.1.1 release will answer.

## Citation

See `CITATION.cff`.

## License

Apache-2.0.
