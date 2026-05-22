# Contributing

Thanks for your interest in `recurrentlens`. The library is in early alpha (v0.1.0); contributions are very welcome.

## Setup

```bash
git clone https://github.com/hinanohart/recurrentlens.git
cd recurrentlens
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest -ra                           # full suite
pytest -ra -k smoke                  # smoke only
RECURRENTLENS_SKIP_GPU_TESTS=1 pytest -ra   # CPU-only (CI default)
```

## Style

- `ruff format`
- `ruff check --fix`
- `pyright` (standard mode)

Pre-commit hooks enforce all three plus a `no-secrets` grep gate.

## High-leverage contributions for v0.1.0 → v0.1.1

1. **Pretrained SAE artifacts**: run `notebooks/03_train_mamba130m_sae.ipynb` on Colab T4, then `recurrentlens.hub.push_sae(...)` to `hinanohart/recurrentlens-mamba130m-L{2,6,10}-sae`. PR adding the Hub IDs to README.
2. **Hook sites for other SSMs**: RWKV-7, Jamba block-local hooks are v0.2 but design notes in `docs/hooks_design.md` welcome.
3. **Feature explorer richness**: v0.1.0 ships a minimal HTML table; richer interactivity (highlighting, k-way comparison) for v0.1.x is open.

## v0.2 roadmap

- `extract_circuit` (recurrent-state-aware patching)
- `diff.compare(sae_mamba, sae_gpt2)`
- Falcon-Mamba-7B Hub registry
- RWKV-7 / Jamba support

## Code of conduct

Be kind. Disagree about technical claims with citations.
