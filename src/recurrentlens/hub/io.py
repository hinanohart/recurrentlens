"""Push/pull SAE artifacts to the HuggingFace Hub.

Phase 8. Authentication is delegated to ``huggingface_hub``: set ``HF_TOKEN``
in your environment, or run ``huggingface-cli login`` before calling
``push_sae``. The token value never passes through the recurrentlens code
path — we only pass ``token=True`` to let huggingface_hub read its own cache.
"""

from __future__ import annotations

import os
from pathlib import Path

from recurrentlens.sae.base import BaseSAE


def load_sae(
    repo_id: str, filename: str = "sae.safetensors", revision: str | None = None
) -> BaseSAE:
    """Download and load a trained SAE from a Hub repo."""
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    return BaseSAE.load(local_path)


def push_sae(
    sae: BaseSAE,
    repo_id: str,
    filename: str = "sae.safetensors",
    private: bool = False,
    create_repo: bool = True,
    commit_message: str | None = None,
) -> str:
    """Upload a trained SAE to a Hub repo and return its URL.

    Requires ``HF_TOKEN`` env var or a prior ``huggingface-cli login``.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    if create_repo:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=private,
        )

    tmp_dir = Path(os.getenv("RECURRENTLENS_HUB_TMP", "/tmp")) / repo_id.replace("/", "_")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_path = tmp_dir / filename
    sae.save(str(local_path))

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message or f"Upload {sae.variant} SAE for {sae.model_id}",
    )
    return f"https://huggingface.co/{repo_id}"
