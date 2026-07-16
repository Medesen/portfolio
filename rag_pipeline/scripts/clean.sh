#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (script lives in scripts/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Sanity check: run from the project root that has main.py and config
if [[ ! -f "main.py" || ! -f "config/config.yaml" ]]; then
  echo "❌ Please run from the project root (main.py + config/config.yaml must exist)."
  exit 1
fi

echo "==> Deep cleaning generated artifacts"

# Only generated paths – DO NOT include source data/corpus or code/config
TARGET_DIRS=(
  "data/processed"
  "data/state"
  "data/vector_store"
  "logs"
)

rm_rf() {
  local p="$1"
  if [[ -e "$p" || -L "$p" ]]; then
    echo "🧹 Removing $p"
    # No silent sudo escalation: if Docker created root-owned files here,
    # say so and let the user decide.
    if ! rm -rf -- "$p" 2>/dev/null; then
      echo "⚠️  Could not remove $p (likely root-owned files from Docker)."
      echo "   Re-run as: sudo rm -rf -- \"$p\""
    fi
  else
    # Ensure parent is not left as an empty dir we want to keep
    true
  fi
}

# Remove generated directories (and their contents)
for d in "${TARGET_DIRS[@]}"; do
  rm_rf "$d"
done

# If top-level 'data' becomes empty after removing generated subdirs, leave it alone.
# (Your source corpus may live under data/corpus/, so we do NOT delete 'data' itself.)

# Dev caches (best-effort; ignore errors)
echo "🧼 Removing Python caches"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Remove model-cache named volume(s), including the legacy name.
# Compose usually names them "<project>_huggingface-cache".
echo "🗑  Removing model cache volumes (if any)"
VOLS="$(docker volume ls -q | grep -E '(^|_)(huggingface-cache|sentence-transformers-cache)$' || true)"
if [[ -n "${VOLS}" ]]; then
  # Try to remove all matches
  echo "${VOLS}" | xargs -r docker volume rm >/dev/null || true
  echo "   Removed: ${VOLS}"
else
  echo "   No model cache volumes found."
fi

# Remove Ollama models named volume(s).
# Compose usually names it "<project>_ollama-models".
echo "🗑  Removing Ollama models volumes (if any)"
OLLAMA_VOLS="$(docker volume ls -q | grep -E '(^|_)ollama-models$' || true)"
if [[ -n "${OLLAMA_VOLS}" ]]; then
  # Try to remove all matches
  echo "${OLLAMA_VOLS}" | xargs -r docker volume rm >/dev/null || true
  echo "   Removed: ${OLLAMA_VOLS}"
else
  echo "   No ollama-models volumes found."
fi

echo "✅ Clean complete."
