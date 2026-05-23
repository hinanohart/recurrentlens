# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.0.post1 | ✅ (current) |
| 0.1.0 | ❌ (superseded by 0.1.0.post1, which fixes correctness/XSS bugs) |

All v0.1.x releases prior to **0.1.0.post1** carry an HTML XSS vector in
`viz.feature_explorer` (unescaped user-controlled fields). Upgrade by:

```bash
uv pip install -U recurrentlens
```

or pin to `recurrentlens>=0.1.0.post1` in your project metadata.

## Reporting a vulnerability

Please report security issues **privately** rather than opening a public
GitHub issue. Use **GitHub's "Report a vulnerability"** form on this
repository's Security tab, or open a [private security advisory](https://github.com/hinanohart/recurrentlens/security/advisories/new).

For sensitive issues that require coordination with HuggingFace Hub (e.g.,
malicious SAE artifacts staged under a Hub repo), please also alert
HuggingFace via their standard responsible-disclosure channel.

## Scope

In scope:

- Code execution via SAE artifacts loaded from the Hub (`hub.load_sae`).
- HTML injection / XSS in `viz.feature_explorer` outputs.
- Path traversal in `ActivationCache` backends or `hub.push_sae` upload paths.
- Secret leakage from environment variables, token files, or logs.
- Dependency CVEs that affect the documented public API.

Out of scope:

- Vulnerabilities in third-party packages (`torch`, `transformers`,
  `huggingface_hub`, `mamba-ssm`, etc.) — please report those upstream.
- Issues that require the attacker to already control the user's Python
  environment (e.g., a malicious site-packages install).
- Denial-of-service via legitimate large activations / large vocabularies.

## Hardening notes

- The `[mamba]` extra installs `mamba-ssm`, which compiles CUDA kernels at
  install time. Build only from trusted environments.
- `hub.load_sae` validates safetensors metadata against tensor shapes
  (since 0.1.0.post1) but **does not** authenticate the publisher of a
  Hub repository. Treat unknown SAEs as untrusted code-adjacent artifacts.
- `feature_explorer.save()` writes a self-contained HTML file. Open the
  file in a browser only if you trust the source of the SAE and the
  source of the `contexts` strings (since 0.1.0.post1 they are HTML-escaped,
  closing the XSS vector against text-based payloads).

## Acknowledgements

The 0.1.0.post1 hotfix (XSS + JumpReLU STE + CE-metric + L0 + load_sae
validation) was driven by a multi-agent audit run shortly after v0.1.0.
See [CHANGELOG.md](./CHANGELOG.md) for the full fix list.
