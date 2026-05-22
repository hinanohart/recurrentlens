"""Sparse autoencoder implementations: vanilla L1, TopK, JumpReLU."""

from recurrentlens.sae.base import BaseSAE
from recurrentlens.sae.cache import ActivationCache
from recurrentlens.sae.variants import JumpReLUSAE, TopKSAE, VanillaSAE, build_sae

__all__ = [
    "BaseSAE",
    "VanillaSAE",
    "TopKSAE",
    "JumpReLUSAE",
    "build_sae",
    "ActivationCache",
]
