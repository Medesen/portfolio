"""Distil RetailRocket's item-property change-logs into one small bundled file.

The raw property files (`item_properties_part1.csv`, `part2.csv`) are ~470 MB
combined — far too large to ship, and 99% of it is hashed property tokens this
project never uses. This script keeps only the two properties Stage 2 needs for
item content features (`categoryid` and `available`) and writes them, gzipped, to
`data/raw/item_properties.csv.gz` (~a few MB).

IMPORTANT — this script is committed for provenance but is NOT run in CI or by
`make reproduce`: it needs the raw download, which is not bundled. The bundled
`item_properties.csv.gz` is its output. `data/raw/README.md` records the source
URL and the sha256 of each raw file so this derived table is reproducible.

The timestamp column is KEPT ON PURPOSE. Property values change over time; taking
an item's *current* category would leak a post-cutoff value into training. Stage 2
takes an as-of-cutoff snapshot (most recent value with timestamp <= cutoff), which
needs the full change history, not a collapsed single value.

Usage:
    python data/prepare_item_properties.py --raw-dir /path/to/extracted/csvs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEEP_PROPERTIES = {"categoryid", "available"}
RAW_FILES = ("item_properties_part1.csv", "item_properties_part2.csv")
OUT = Path(__file__).resolve().parent / "raw" / "item_properties.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory holding the extracted item_properties_part{1,2}.csv files",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    frames = []
    for name in RAW_FILES:
        path = args.raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; extract the Kaggle archive first")
        print(f"reading {name} ...", flush=True)
        df = pd.read_csv(path, usecols=["timestamp", "itemid", "property", "value"])
        kept = df[df["property"].isin(KEEP_PROPERTIES)]
        print(f"  {len(df):,} rows -> {len(kept):,} kept ({sorted(KEEP_PROPERTIES)})")
        frames.append(kept)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["itemid", "property", "timestamp"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, compression="gzip")
    print(
        f"\nwrote {args.out} — {len(out):,} property snapshots for "
        f"{out['itemid'].nunique():,} items ({args.out.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
