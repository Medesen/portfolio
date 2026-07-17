# DATASET CARD — scikit-learn 1.7.2 

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
- Renamed the folder to `scikit-learn-1.7.2-docs/`.
- **Kept**:
- `modules/` (all top-level user guide pages)
- `modules/generated/` **only** files starting with:
  `sklearn.linear_model.
    sklearn.preprocessing.
    sklearn.model_selection.
    sklearn.metrics.
    sklearn.impute.`
- `api/` (entire)
- `auto_examples/` subfolders: impute, linear_model, model_selection, preprocessing
- **Removed** everything else:

  - _downloads
  - _images
  - _sources
  - _sphinx_design_static
  - _static
  - binder
  - computing
  - datasets
  - developers
  - lite
  - notebooks
  - testimonials
  - tutorial
  - whats_new

## Intended use
- Retrieval-augmented Q&A over core scikit-learn topics (preprocessing, model selection,
linear models, basic metrics/imputation) with short, grounded answers and citations.

## Notes and limitations
- This is a *local snapshot* for reproducible evaluation; it does not auto-update.
- The keep/delete patterns are tuned for scikit-learn 1.7.2; future releases may
add or rename pages. Re-run this same script to regenerate a consistent corpus.

## Provenance
- Version: 1.7.2 
- Processed at: 2026-07-17 11:09:44 UTC
- Script: scripts/prune_sklearn_corpus.py (deterministic rules as above)
