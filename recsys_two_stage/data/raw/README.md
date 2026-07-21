# Dataset: RetailRocket Recommender System Dataset

Real e-commerce behavioural data from an anonymous online retailer: view /
add-to-cart / transaction events over 4.5 months, plus item property change-logs and
a category tree. Published to motivate recommender-systems research on implicit
feedback.

| | |
|---|---|
| **Source** | Retailrocket, *Retailrocket recommender system dataset*, published by Roman Zykov on Kaggle: <https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset> |
| **Retrieved** | 2026-07-20 |
| **License** | **CC BY-NC-SA 4.0** — <https://creativecommons.org/licenses/by-nc-sa/4.0/>. Free to share and adapt for **non-commercial** use with **attribution**; adaptations must carry the **same licence** (ShareAlike). |

## Bundled files

| File | Size | Contents |
|---|---|---|
| `events.csv.gz` | 32 MB | The full event log, gzipped verbatim from the source (no rows removed). Filtering happens in code at load time, not by shipping a pre-filtered file. |
| `item_properties.csv.gz` | 8 MB | **Derived.** The two properties this project uses (`categoryid`, `available`), distilled from the ~470 MB raw property files by `data/prepare_item_properties.py`. |
| `category_tree.csv.gz` | 6 KB | Category → parent-category hierarchy, gzipped verbatim. |

The raw `item_properties_part1.csv` (484 MB) and `item_properties_part2.csv` (409 MB)
are **not** bundled — too large, and 99% is hashed tokens unused here. The
distillation script is committed for provenance but is **not** run in CI (it needs
the raw download). To regenerate the derived file, download the Kaggle archive and:

```bash
python data/prepare_item_properties.py --raw-dir /path/to/extracted/csvs
```

### Checksums (sha256)

Of the **bundled** files (verify with `gunzip -c <file> | sha256sum` for the two
verbatim files):

| File | Decompressed sha256 |
|---|---|
| `events.csv` (inside `events.csv.gz`) | `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7` |
| `category_tree.csv` (inside `category_tree.csv.gz`) | `94e865eb0a3d48cbbfe3b79079018dd92509315c88f5fd8d00d0b4b5af434f5b` |

Of the **raw** source files the derived `item_properties.csv.gz` was distilled from
(for reproducibility of the derivation):

| Raw file | sha256 |
|---|---|
| `item_properties_part1.csv` | `30aad5aeca58b2dc27dcc73e1708565f5818e45adb3eb57401f91e87355b0b81` |
| `item_properties_part2.csv` | `d5e7d1a91dc40f522aeb596b267e6c87d8aed689a7192d12369cfb165eb987e5` |

## Columns

### `events.csv.gz`

| Column | Meaning |
|---|---|
| `timestamp` | Event time, milliseconds since Unix epoch (UTC) |
| `visitorid` | Anonymous visitor id (a cookie, not a persistent account) |
| `event` | `view`, `addtocart`, or `transaction` |
| `itemid` | Item interacted with |
| `transactionid` | Populated only on `transaction` events; null otherwise (99.2% of rows) |

2,756,101 events (2,664,312 views / 69,332 add-to-carts / 22,457 transactions),
1,407,580 visitors, 235,061 items, 2015-05-03 to 2015-09-18. All values are hashed
by the publisher for confidentiality.

### `item_properties.csv.gz` (derived)

| Column | Meaning |
|---|---|
| `timestamp` | When the property value was recorded (ms since epoch). **Kept on purpose** — property values change over time, and Stage 2 takes an as-of-cutoff snapshot to avoid leaking future values into training. |
| `itemid` | Item |
| `property` | `categoryid` or `available` (the two retained from the raw files) |
| `value` | Property value. `categoryid` is an integer id joinable to `category_tree`; `available` is 0/1. |

### `category_tree.csv.gz`

| Column | Meaning |
|---|---|
| `categoryid` | Category id |
| `parentid` | Parent category id (null at the root) |

## Why this dataset

RetailRocket was chosen over MovieLens — the field's usual benchmark — for three
reasons. Its licence permits redistribution, so the data ships in the repository and
the project keeps the portfolio's clone-and-run promise. It is real e-commerce
behaviour (views, add-to-carts, purchases) with genuine timestamps, so a time-based
split reflects how a deployed system actually works. And it carries product
attributes, which is the only reason the cold-start comparison in Stage 2 is possible
at all. The cost is comparability: MovieLens results can be checked against published
numbers, and these cannot — a trade made deliberately, in favour of realism and a
licence that allows the data to be shipped.

## Citation

Zykov, R. / Retailrocket. *Retailrocket recommender system dataset.* Kaggle.
<https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset>
