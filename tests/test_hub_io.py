"""Regression tests for hub/io.py filename validation (path-traversal fix)."""

from __future__ import annotations

import pytest

from recurrentlens.hub.io import _validate_filename


@pytest.mark.parametrize(
    "bad",
    [
        "../evil.safetensors",
        "../../etc/passwd",
        "/absolute/path.safetensors",
        "sub/dir/file.safetensors",
        "sub\\file.safetensors",
        "..",
        "C:\\Windows\\System32\\evil",
    ],
)
def test_validate_filename_rejects_traversal(bad: str) -> None:
    """Path-traversal and absolute filenames must be rejected."""
    with pytest.raises(ValueError):
        _validate_filename(bad)


@pytest.mark.parametrize(
    "good",
    [
        "sae.safetensors",
        "my_sae_v2.safetensors",
        "model-checkpoint.pt",
        "sae",
    ],
)
def test_validate_filename_accepts_safe_basename(good: str) -> None:
    """Plain basenames with no separators or '..' must be accepted."""
    _validate_filename(good)  # must not raise
