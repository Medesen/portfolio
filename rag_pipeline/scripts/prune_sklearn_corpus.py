#!/usr/bin/env python3
"""
Prepare a reproducible, slimmed-down scikit-learn documentation corpus for RAG.

Usage:
  python scripts/prune_sklearn_corpus.py \
      --root data/corpus \
      --input-name scikit-learn-docs \
      [--suffix stable|dev|none] \
      [--dry-run]

Assumptions:
- You've unzipped the official scikit-learn docs ZIP so that it created
  a directory called 'scikit-learn-docs/' under --root (default: data/corpus).
- Script will rename it to 'scikit-learn-<version>[-<suffix>]-docs/' and prune.

This script is idempotent: if already renamed, you can pass the renamed folder
with --input-name and it will still prune + (re)write the DATASET_CARD.md.
"""

from __future__ import annotations
import argparse
import datetime as dt
import re
import shutil
from pathlib import Path

KEEP_API = True  # keep entire api/ (small, useful)
# Keep selected auto_examples subfolders (the "keep-a-bit-more" variant)
KEEP_AUTO_EXAMPLES = {"preprocessing", "model_selection", "linear_model", "impute"}

# Keep only these namespaces under modules/generated
KEEP_GENERATED_PREFIXES = (
    "sklearn.linear_model.",
    "sklearn.preprocessing.",
    "sklearn.model_selection.",
    "sklearn.metrics.",
    "sklearn.impute.",
)

# Top-level directories to delete entirely if present
DELETE_TOP_LEVEL_DIRS = {
    "_images",
    "_static",
    "_sphinx_design_static",
    "_downloads",
    "_sources",
    "lite",
    "developers",
    "datasets",
    "computing",
    "notebooks",
    "testimonials",
    "tutorial",
    "whats_new",
    "binder",
}

CARD_TEMPLATE = """\
# DATASET CARD — scikit-learn {version} {label}

**Source**: Official scikit-learn HTML documentation (offline bundle).  
**Project**: scikit-learn — Machine Learning in Python.  
**Homepage**: https://scikit-learn.org/  
**Versions page (offline bundles)**: https://scikit-learn.org/dev/versions.html  
**License**: BSD 3-Clause (documentation). See: https://scikit-learn.org/stable/about.html#license

## Acquisition
- Download the official HTML docs ZIP for the chosen version from the versions page.
- Unzip to a folder named `scikit-learn-docs/`.
- Run the prep script in this repository:

## Preprocessing performed (this script)
- Renamed the folder to `scikit-learn-{version}{dash_label}-docs/`.
- **Kept**:
- `modules/` (all top-level user guide pages)
- `modules/generated/` **only** files starting with:
  `{prefixes}`
- `api/` (entire)
- `auto_examples/` subfolders: {examples}
- **Removed** everything else:
{deleted}

## Intended use
- Retrieval-augmented Q&A over core scikit-learn topics (preprocessing, model selection,
linear models, basic metrics/imputation) with short, grounded answers and citations.

## Notes and limitations
- This is a *local snapshot* for reproducible evaluation; it does not auto-update.
- The keep/delete patterns are tuned for scikit-learn {version}; future releases may
add or rename pages. Re-run this same script to regenerate a consistent corpus.

## Provenance
- Version: {version} {label}
- Processed at: {timestamp} UTC
- Script: scripts/prune_sklearn_corpus.py (deterministic rules as above)
"""

def detect_version_and_label(doc_root: Path, user_suffix: str | None) -> tuple[str, str | None]:
  """
  Extract VERSION from _static/documentation_options.js if present.
  Try to infer 'stable'/'dev' label from index.html canonical link; fall back to user_suffix.
  """
  version = None
  # 1) Try documentation_options.js
  js_path = doc_root / "_static" / "documentation_options.js"
  if js_path.exists():
      m = re.search(r"VERSION:\s*'([^']+)'", js_path.read_text(encoding="utf-8", errors="ignore"))
      if m:
          version = m.group(1).strip()

  # 2) Fallback: parse index.html <title> or elsewhere
  if version is None and (doc_root / "index.html").exists():
      html = (doc_root / "index.html").read_text(encoding="utf-8", errors="ignore")
      # Common patterns:
      #   'scikit-learn 1.7.2 documentation'
      #   '... — scikit-learn 1.7.2 documentation'
      m = re.search(r"scikit-learn\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)\s+documentation", html, re.I)
      if m:
          version = m.group(1)

  if version is None:
      raise RuntimeError("Could not detect scikit-learn version; ensure the docs are intact.")

  # Detect label from canonical link
  label = None
  if (doc_root / "index.html").exists():
      html = (doc_root / "index.html").read_text(encoding="utf-8", errors="ignore")
      c = re.search(r'href="https?://scikit-learn\.org/([^"/]+)/?"', html)
      if c:
          segment = c.group(1).lower()
          if segment in {"stable", "dev"}:
              label = segment

  # User override
  if user_suffix and user_suffix.lower() in {"stable", "dev", "none"}:
      label = None if user_suffix.lower() == "none" else user_suffix.lower()

  return version, label


def rename_root(root: Path, input_name: str, version: str, label: str | None, dry_run: bool) -> Path:
  src = root / input_name
  if not src.exists():
      raise FileNotFoundError(f"Input folder not found: {src}")

  label_part = f"-{label}" if label else ""
  dst_name = f"scikit-learn-{version}{label_part}-docs"
  dst = root / dst_name

  if src.resolve() == dst.resolve():
      print(f"[rename] Already named {dst_name}")
      return dst

  if dst.exists():
      # If the target exists, assume prior run; switch to pruning within that folder
      print(f"[rename] Target already exists: {dst_name} (skipping rename)")
      return dst

  print(f"[rename] {src.name} -> {dst_name}")
  if not dry_run:
      src.rename(dst)
  return dst


def delete_dir(path: Path, dry_run: bool):
  if path.exists():
      print(f"[delete] {path.relative_to(path.parents[1])}")
      if not dry_run:
          shutil.rmtree(path)


def prune_top_level(doc_root: Path, dry_run: bool):
  for name in list(DELETE_TOP_LEVEL_DIRS):
      delete_dir(doc_root / name, dry_run)


def prune_auto_examples(doc_root: Path, dry_run: bool):
  ae = doc_root / "auto_examples"
  if not ae.exists():
      return
  # Remove everything except the selected subfolders
  for child in ae.iterdir():
      if child.is_dir():
          if child.name not in KEEP_AUTO_EXAMPLES:
              delete_dir(child, dry_run)
      else:
          # remove stray files at root of auto_examples
          print(f"[delete] auto_examples/{child.name}")
          if not dry_run:
              child.unlink(missing_ok=True)


def prune_generated(doc_root: Path, dry_run: bool):
  gen = doc_root / "modules" / "generated"
  if not gen.exists():
      return
  for f in gen.iterdir():
      if f.is_file():
          keep = any(f.name.startswith(prefix) for prefix in KEEP_GENERATED_PREFIXES)
          if not keep:
              print(f"[delete] modules/generated/{f.name}")
              if not dry_run:
                  f.unlink()
      else:
          # Shouldn’t usually have subdirs here, but delete if present
          delete_dir(f, dry_run)


def maybe_delete_api(doc_root: Path, dry_run: bool):
  if KEEP_API:
      return
  delete_dir(doc_root / "api", dry_run)


def write_dataset_card(doc_root: Path, version: str, label: str | None, dry_run: bool):
  label_str = label or ""
  dash_label = f"-{label}" if label else ""
  card_path = doc_root / "DATASET_CARD.md"
  content = CARD_TEMPLATE.format(
      version=version,
      label=label_str,
      dash_label=dash_label,
      prefixes="\n    ".join(KEEP_GENERATED_PREFIXES),
      examples=", ".join(sorted(KEEP_AUTO_EXAMPLES)),
      deleted="\n  - " + "\n  - ".join(sorted(DELETE_TOP_LEVEL_DIRS)),
      timestamp=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
  )
  print(f"[write] {card_path.relative_to(doc_root.parents[0])}")
  if not dry_run:
      card_path.write_text(content, encoding="utf-8")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", type=Path, default=Path("data/corpus"), help="Parent directory containing the docs folder.")
  ap.add_argument("--input-name", type=str, default="scikit-learn-docs", help="Name of the unpacked docs directory.")
  ap.add_argument("--suffix", type=str, default=None, help="Optional override for label suffix: stable|dev|none.")
  ap.add_argument("--dry-run", action="store_true", help="Show actions without changing files.")
  args = ap.parse_args()

  doc_root = args.root / args.input_name
  if not doc_root.exists():
      raise SystemExit(f"Could not find docs at: {doc_root}")

  # Detect version/label BEFORE deleting _static
  version, label = detect_version_and_label(doc_root, args.suffix)

  # Rename root if needed
  doc_root = rename_root(args.root, args.input_name, version, label, args.dry_run)

  # Prune
  prune_top_level(doc_root, args.dry_run)
  prune_auto_examples(doc_root, args.dry_run)
  prune_generated(doc_root, args.dry_run)
  maybe_delete_api(doc_root, args.dry_run)

  # Write dataset card
  write_dataset_card(doc_root, version, label, args.dry_run)

  print("\n[done] Corpus prepared.")
  print(f"       Location: {doc_root}")
  print("       Wrote: DATASET_CARD.md")


if __name__ == "__main__":
  main()
