#!/usr/bin/env bash
# Grep gate for accidentally committed tokens (R11 compliance).
# Looks for patterns matching common API token shapes in tracked files.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Patterns: GitHub PAT, HF token, OpenAI key, AWS access key.
PATTERNS=(
  'gh[oprsu]_[A-Za-z0-9]{36,}'
  'github_pat_[A-Za-z0-9_]{82,}'
  'hf_[A-Za-z0-9]{34,}'
  'sk-[A-Za-z0-9]{20,}'
  'AKIA[0-9A-Z]{16}'
)

EXCLUDE_RE='\.(safetensors|pt|zarr|ipynb_checkpoints)|__pycache__|^\.git/|scripts/check_no_secrets\.sh'

found=0
for pat in "${PATTERNS[@]}"; do
  if matches=$(git ls-files | grep -Ev "$EXCLUDE_RE" | xargs -d '\n' grep -EHn "$pat" 2>/dev/null || true); then
    if [ -n "$matches" ]; then
      echo "Possible secret found (pattern: $pat):"
      echo "$matches"
      found=1
    fi
  fi
done

if [ "$found" -eq 1 ]; then
  echo ""
  echo "Refusing to commit. Remove the secret, rotate it, and try again."
  exit 1
fi

echo "no-secrets grep gate: clean"
