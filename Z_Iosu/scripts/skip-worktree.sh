#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-apply}"
if [[ "$ACTION" != "apply" && "$ACTION" != "clear" ]]; then
  echo "Usage: $0 [apply|clear]" >&2
  exit 1
fi

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "$ROOT" ]]; then
  echo "Not inside a git repo" >&2
  exit 1
fi
cd "$ROOT"

if [[ ! -d Z_Iosu ]]; then
  echo "Z_Iosu not found" >&2
  exit 0
fi

mapfile -t FILES < <(git ls-files Z_Iosu)
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No tracked files under Z_Iosu"
  exit 0
fi

if [[ "$ACTION" == "apply" ]]; then
  for f in "${FILES[@]}"; do git update-index --skip-worktree -- "$f"; done
  echo "Applied skip-worktree on Z_Iosu"
else
  for f in "${FILES[@]}"; do git update-index --no-skip-worktree -- "$f"; done
  echo "Cleared skip-worktree on Z_Iosu"
fi

# Verify
COUNT=$(git ls-files -v Z_Iosu | grep -cE '^[sS] ' || true)
if [[ "$ACTION" == "apply" ]]; then
  echo "Files with 'S' (skip-worktree): $COUNT"
else
  echo "Files with 'S' after clear: $COUNT"
fi
