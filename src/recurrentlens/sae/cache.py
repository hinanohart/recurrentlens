"""Activation cache backends: ram / mmap / zarr.

Phase 4 minimal implementation. The cache is a fixed-shape (N, d_in) float
buffer that the trainer fills from forward-hooked activations and reads in
shuffled batches.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from recurrentlens.core.types import CacheBackend


class ActivationCache:
    """A simple append-only cache with three storage backends.

    Parameters
    ----------
    capacity : int
        Total number of activation rows.
    d_in : int
        Per-row dimension.
    backend : "ram" | "mmap" | "zarr"
    path : optional pathlib.Path or str (required for mmap/zarr).
    dtype : numpy dtype, default float32.
    """

    def __init__(
        self,
        capacity: int,
        d_in: int,
        backend: CacheBackend = "mmap",
        path: str | Path | None = None,
        dtype: Any = np.float32,
    ):
        self.capacity = int(capacity)
        self.d_in = int(d_in)
        self.backend = backend
        self.dtype = np.dtype(dtype)
        self._size = 0

        if backend == "ram":
            self._store = np.zeros((self.capacity, self.d_in), dtype=self.dtype)
            self._path: Path | None = None
        elif backend == "mmap":
            if path is None:
                fd, tmp = tempfile.mkstemp(suffix=".memmap")
                os.close(fd)
                path = tmp
            self._path = Path(path)
            if self._path.exists() and self._path.stat().st_size > 0:
                warnings.warn(
                    f"ActivationCache mmap path {self._path} exists and will be "
                    "truncated to a fresh (capacity, d_in) buffer. Pass a new "
                    "path or backend='ram' to preserve existing data.",
                    UserWarning,
                    stacklevel=2,
                )
            self._store = np.memmap(
                self._path, mode="w+", shape=(self.capacity, self.d_in), dtype=self.dtype
            )
        elif backend == "zarr":
            import zarr

            if path is None:
                raise ValueError("zarr backend requires `path`")
            self._path = Path(path)
            self._store = zarr.open(
                str(self._path),
                mode="w",
                shape=(self.capacity, self.d_in),
                chunks=(min(self.capacity, 65536), self.d_in),
                dtype=str(self.dtype),
            )
        else:
            raise ValueError(f"unknown backend: {backend!r}")

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    def append(self, batch: Any) -> int:
        """Append a (B, d_in) batch. Returns number of rows actually written."""
        arr = np.asarray(batch, dtype=self.dtype)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        assert arr.ndim == 2 and arr.shape[1] == self.d_in, (
            f"expected (*, {self.d_in}), got {arr.shape}"
        )
        room = self.capacity - self._size
        to_write = min(arr.shape[0], room)
        if to_write <= 0:
            return 0
        self._store[self._size : self._size + to_write] = arr[:to_write]
        self._size += to_write
        return to_write

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return a uniform random batch of size ``batch_size`` (with replacement)."""
        if self._size == 0:
            raise RuntimeError("cache is empty; nothing to sample")
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self._size, size=batch_size)
        return np.asarray(self._store[idx])

    def close(self) -> None:
        if self.backend == "mmap" and hasattr(self._store, "flush"):
            self._store.flush()
