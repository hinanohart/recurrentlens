#!/usr/bin/env python3
"""Smoke training script for Mamba-130M SAE.

Designed for Colab T4 / RTX 3090+ class machines. CPU-runnable but slow
(< 1M tokens recommended for CPU smoke).

Example::

    python scripts/train_mamba130m_smoke.py \\
        --model state-spaces/mamba-130m-hf \\
        --layer 6 --variant topk --d-sae 16384 --n-tokens 5_000_000 \\
        --save-to sae_L6.safetensors
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="state-spaces/mamba-130m-hf")
    p.add_argument("--hook-site", default="out_proj_out", choices=["out_proj_out", "ssm_h_t"])
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--variant", default="topk", choices=["vanilla", "topk", "jumprelu"])
    p.add_argument("--d-sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32, help="TopK k (TopK variant only)")
    p.add_argument("--n-tokens", type=int, default=5_000_000)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--cache-backend", default="mmap", choices=["ram", "mmap", "zarr"])
    p.add_argument("--cache-path", default=None)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--save-to", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--push-to-hub", default=None, help="if set, repo_id to push the trained SAE")
    args = p.parse_args()

    import recurrentlens as rl

    model = rl.load_model(args.model, device=args.device)
    sae = rl.train_sae(
        model,
        hook_site=args.hook_site,
        layer=args.layer,
        variant=args.variant,
        d_sae=args.d_sae,
        dataset=args.dataset,
        n_tokens=args.n_tokens,
        n_steps=args.n_steps,
        cache_backend=args.cache_backend,
        cache_path=args.cache_path,
        save_to=args.save_to,
        k=args.k,
    )
    print(f"saved SAE to {args.save_to}")

    if args.push_to_hub:
        url = rl.hub.push_sae(sae, repo_id=args.push_to_hub)
        print(f"pushed to {url}")


if __name__ == "__main__":
    main()
