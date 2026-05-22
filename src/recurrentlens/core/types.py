"""Type aliases and constants used across recurrentlens."""

from __future__ import annotations

from typing import Literal, TypeAlias

HookSite: TypeAlias = Literal["out_proj_out", "ssm_h_t"]
SAEVariant: TypeAlias = Literal["vanilla", "topk", "jumprelu"]
CacheBackend: TypeAlias = Literal["ram", "mmap", "zarr"]

DEFAULT_HOOK_SITE: HookSite = "out_proj_out"
DEFAULT_SAE_VARIANT: SAEVariant = "topk"
DEFAULT_CACHE_BACKEND: CacheBackend = "mmap"

# Architecture decision: out_proj_out is the residual-stream analog and the
# safest bet for monosemantic features. ssm_h_t (recurrent state) is the
# experimental, paper-worthy site that we expose but do not default to.
HOOK_SITES: tuple[HookSite, ...] = ("out_proj_out", "ssm_h_t")
SAE_VARIANTS: tuple[SAEVariant, ...] = ("vanilla", "topk", "jumprelu")
CACHE_BACKENDS: tuple[CacheBackend, ...] = ("ram", "mmap", "zarr")
