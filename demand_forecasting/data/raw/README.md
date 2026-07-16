# Dataset: Hierarchical Sales Data (Italian pasta sales)

Daily sales quantities and promotion flags for 118 pasta SKUs (4 brands) sold by
an Italian grocery retailer, 2014-01-02 to 2018-12-31 (the store is closed on
New Year's Day, so the panel starts on Jan 2).

| | |
|---|---|
| **File** | `hierarchical_sales_data.csv` (872 KB, bundled in this repository) |
| **Shape** | 1,798 daily rows × 237 columns (`DATE` + 118 `QTY_B{brand}_{item}` + 118 `PROMO_B{brand}_{item}`) |
| **`QTY_*`** | Units sold that day for one SKU (integer) |
| **`PROMO_*`** | 1 if that SKU was on promotion that day, 0 otherwise (binary) |
| **Source** | Mendeley Data: <https://data.mendeley.com/datasets/njdkntcpc9/1> (also listed as UCI ML Repository dataset #611, "Hierarchical Sales Data") |
| **Retrieved** | 2026-07-03, sha256 `0dfd4a5bf801bf79ddb5fb1f6bc8d023487c36dd6a2a70b312cdfa9fa6568d83` |
| **License** | UCI lists CC BY 4.0; the Mendeley record lists CC BY-NC 3.0. Both permit redistribution with attribution; this repository is non-commercial, so the stricter reading is satisfied as well. |

## Citation

Mancuso, P., Piccialli, V., & Sudoso, A. M. (2021). *A machine learning approach
for forecasting hierarchical time series.* Expert Systems with Applications,
182, 115102. <https://doi.org/10.1016/j.eswa.2021.115102>

## Why this dataset

It is, to our knowledge, the only openly licensed retail dataset that combines
(a) several years of **daily** history across many series with (b) an explicit,
per-SKU **promotion flag** — which is what makes both the forecasting comparison
and the promotion-lift analysis in this project possible without any
authentication or licensing caveats for whoever clones this repository.
