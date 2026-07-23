# Customer Churn Prediction - Production ML Pipeline

[![CI Pipeline](https://github.com/Medesen/portfolio/workflows/CI%20-%20End2End%20Churn%20Prediction/badge.svg)](https://github.com/Medesen/portfolio/actions/workflows/ci-end2end-churn.yml)

A production-oriented ML service that predicts customer churn for a telecom dataset. I built this to demonstrate end-to-end ML engineering: from training with hyperparameter tuning to serving via REST API, with MLflow experiment tracking, drift detection, monitoring dashboards, and a CI/CD pipeline. Everything runs in Docker with zero local setup required.

## Summary

**Domain:** Telecom customer churn prediction  
**Dataset:** Telco Customer Churn (7,043 customers, 19 features)  
**Task:** Binary classification  
**Best Performance:** ROC AUC 0.837 (held-out test set)  
**Supported Models:** Random Forest, XGBoost, Logistic Regression  
**Tech Stack:** FastAPI, MLflow, Prometheus/Grafana, Docker, Kubernetes  
**Test Coverage:** 212 tests, 91% full-suite coverage of src/: the serving (`src/api`) and training (`src/training`) packages are inside the measured universe, and CI enforces the number with a full-suite coverage gate (floor 88%)

**Model Performance (held-out test set, F1-tuned threshold ≈ 0.33):**
- ROC AUC: 0.837
- Precision (Churn): 53%
- Recall (Churn): 74%
- F1: 0.62

At the default 0.5 threshold the model trades the other way: higher precision, much lower recall (67% / 48% on the validation set). Both are reported, clearly labelled, in [Performance Metrics](#performance-metrics).

**Business Context:** In churn prediction, false negatives (missed churners) cost significantly more than false positives (unnecessary retention campaigns) due to lost customer lifetime value; industry estimates commonly put the ratio at 10-50x. The model uses threshold tuning to balance these asymmetric costs, prioritising recall while maintaining acceptable precision.

---

## Quick Start (5 Minutes)

**Prerequisites:** Docker Desktop with Compose V2 ([Get Docker](https://docs.docker.com/get-docker/)); 2 CPU cores, 4 GB RAM, ~2 GB disk. This project runs exclusively via Docker: no Python, pip, or conda needed.

```bash
# Clone and enter the project
git clone https://github.com/Medesen/portfolio.git
cd portfolio/end2end_churn

# Automated setup: builds containers (~2-3 min), trains a model (~1-2 min)
make setup

# Then:
make up          # Start API service
make test-api    # Test a prediction
make down        # Stop service
make help        # See all commands
```

**Windows:** `make` is not available by default: either `choco install make`, run `.\setup.ps1`, or use the direct Docker commands. All three options are described in [docs/OPERATIONS.md](docs/OPERATIONS.md#windows-setup-options).

---

## What This Project Demonstrates

The training pipeline handles three algorithms with automated hyperparameter tuning (grid search + cross-validation), logs every run to MLflow, and publishes versioned models with SHA256 integrity checksums and drift-detection reference statistics baked into the metadata.

The API layer is a decomposed FastAPI application (`src/api`): an app factory wires route modules (predict, health, drift, retrain), four middleware layers (request tracing, timeout protection, Prometheus metrics, streaming body-size enforcement), optional constant-time bearer-token auth, and sanitising error handlers that log everything but leak nothing. Drift detection monitors numeric features (mean/std + KS test), categorical features (PSI), and prediction distributions; retraining triggers go through an atomic file-lock reservation so concurrent triggers can't launch duplicate runs.

The monitoring stack uses Prometheus and Grafana with three dashboards (API overview, latency/SLO, ML metrics). Deployment is fully containerised, with Kubernetes manifests including auto-scaling, health probes, and resource limits.

### Key Technical Decisions

**Docker-only execution:** Rather than support both local and containerised workflows, I made Docker mandatory. This eliminates "works on my machine" issues (reviewers run the project with just `make setup`) and enforces production parity: training happens in the same environment where models deploy.

**Threshold optimisation:** The default 0.5 classification threshold ignores both class imbalance and business costs. I implemented F1-optimised thresholds (≈0.33 for this dataset) plus precision-constrained, top-K, and cost-sensitive alternatives; all are computed at training time and stored in model metadata, and the API applies the tuned threshold automatically.

**Entry points as thin shims:** `serve.py` and `train.py` only exist to keep deployment surfaces stable (`uvicorn serve:app`, `python train.py`). The logic lives in `src/api` and `src/training`, where it is imported, typed, and measured by coverage like everything else.

**Gradual typing boundary:** The serving and training packages are mypy-clean under a strict configuration and CI *blocks* on them staying clean. Older analytics modules (~70 remaining errors, mostly ML-library interface friction) run as an advisory burn-down list: the boundary moves as modules are cleaned, and the README doesn't claim more type safety than CI enforces.

**Drift detection trade-offs:** Statistical drift detection (mean/std/KS for numeric, PSI for categorical) rather than model-based approaches: faster and interpretable, but may miss subtle drifts that don't affect distributions. Thresholds are configurable since optimal values depend on business tolerance for false alarms vs missed drift.

---

## Testing

212 tests across unit, integration, and end-to-end tiers cover preprocessing, the training pipeline, model factory, API endpoints (including auth 401 paths and lifespan startup/shutdown), drift detection, threshold tuning, retraining coordination, and monitoring integration. Full-suite coverage is 91% of `src/` (which since the entry-point decomposition includes all serving and training code) and the CI full-coverage job enforces a floor of 88%, so the headline number is verifiable from a green build alone.

```bash
make test               # All tests
make test-coverage      # With coverage report
make test-unit          # Unit tests (fast)
make test-integration   # API integration tests
make test-e2e           # End-to-end workflows
```

Leakage-relevant hygiene: model selection uses the validation split; the held-out test set is evaluated once, at the tuned threshold, and headline numbers come from it. The e2e tests run the real CLI via subprocess (which still counts toward coverage: the subprocess runs under the same interpreter).

**Production considerations:** comprehensive for a portfolio project but not exhaustive. For production I would add property-based testing, chaos tests, and performance regression tests.

---

## Dataset

Kaggle Telco Customer Churn (IBM sample data): 7,043 customers, 19 features (4 numeric + 15 categorical; customerID and the Churn target excluded), ~26.5% churn rate. Feature groups: demographics (gender, SeniorCitizen, Partner, Dependents), services (phone/internet/streaming add-ons), account (tenure, Contract, PaperlessBilling, PaymentMethod, charges).

---

## API

Available at http://localhost:8000 (`GET /docs` for interactive Swagger UI):

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Churn prediction (rate-limited, optional bearer auth) |
| `POST /drift`, `GET /drift/info` | Batch drift analysis and baseline inspection |
| `POST /retrain` | Trigger retraining (atomic rate-limit reservation) |
| `GET /health`, `/healthz`, `/readyz` | Detailed health, liveness, readiness probes |
| `GET /metrics` | Prometheus metrics |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
# {"churn_probability": 0.56, "churn_prediction": "Yes", "risk_level": "Medium", ...}
```

---

## Model Training

```bash
make train             # Random Forest, default config
make train-quick       # Fast iteration
make train-prod        # Extensive search
make train-xgboost     # or train-rf / train-logreg
make compare-models    # Compare all three
make mlflow-ui         # View experiments at http://localhost:5000
```

The pipeline (`src/training/pipeline.py`) loads and preprocesses the data, makes a 3-way split, grid-searches with cross-validation, evaluates on the validation set, tunes the classification threshold, evaluates the held-out test set once at that threshold, computes drift reference statistics, and publishes versioned artifacts, with every run tracked in MLflow (failed runs are marked FAILED, not silently FINISHED).

---

## Project Structure

```
end2end_churn/
├── src/
│   ├── api/                      # FastAPI service package
│   │   ├── app.py                #   app factory + lifespan (model loading)
│   │   ├── routes/               #   predict, health, drift, retrain, info
│   │   ├── middleware.py         #   tracing, timeout, metrics, size limit
│   │   ├── auth.py               #   constant-time bearer-token auth
│   │   ├── retraining.py         #   atomic retrain reservation + task
│   │   ├── errors.py             #   sanitising exception handlers
│   │   ├── settings.py           #   env-derived runtime config
│   │   ├── service.py            #   model cache + prediction logic
│   │   ├── schemas.py            #   Pydantic request/response models
│   │   └── validation.py         #   schema alignment + Pandera checks
│   ├── training/                 # Training pipeline package
│   │   ├── pipeline.py           #   orchestration (main)
│   │   ├── reporting.py          #   evaluation, thresholds, diagnostics
│   │   ├── artifacts.py          #   model/metadata publication, ref stats
│   │   └── mlflow_logging.py     #   MLflow model logging + registry
│   ├── data/                     # Loading & preprocessing
│   ├── models/                   # Pipelines & model factory
│   ├── evaluation/               # Metrics & visualisations
│   ├── utils/                    # Logging, I/O, Prometheus, drift
│   └── config.py                 # Training/service configuration
├── serve.py                      # Thin shim: uvicorn serve:app
├── train.py                      # Thin shim: CLI → src.training.pipeline
├── tests/                        # 212 tests (unit/integration/e2e markers)
├── config/ k8s/ grafana/         # Configs, K8s manifests, dashboards
├── docs/                         # OPERATIONS.md (ops guide), SLO.md
└── Dockerfile, docker-compose.yml, Makefile
```

---

## CI/CD Pipeline

GitHub Actions on every PR/push:

1. **Code Quality**: flake8 (pyflakes class) + black + isort blocking; mypy blocking on the serving/training packages, advisory on the legacy burn-down list
2. **Unit Tests**: enforced coverage floor (unit-only, 50%)
3. **Integration Tests**: API endpoint validation against a freshly trained model
4. **Full-Suite Coverage Gate**: entire suite in one run, floor 88% (verifies the 91% headline)
5. **E2E Tests**: full training workflows via the real CLI
6. **Docker Build**: production image build and smoke test
7. **Security Scan**: Trivy, *advisory*: findings reported to the Security tab but non-blocking (documented in the workflow, with the tightening path)
8. **Load Testing**: Locust run with SLO validation (fails on violations)

View results: [GitHub Actions](https://github.com/Medesen/portfolio/actions/workflows/ci-end2end-churn.yml)

---

## Performance Metrics

**Model performance (held-out test set, F1-tuned threshold ≈ 0.33):**
- ROC AUC: 0.837 | Precision (Churn): 53% | Recall (Churn): 74% | Accuracy: 76%

**For comparison (validation set, default 0.5 threshold):**
- ROC AUC: 0.831 | Precision (Churn): 67% | Recall (Churn): 48% | Accuracy: 80%

Threshold tuning deliberately trades precision and raw accuracy for recall: the right direction when a missed churner costs far more than a wasted retention offer. The model catches 275 of 374 churners in the held-out test set.

**Top predictive features:** tenure (13.9%), TotalCharges (11.9%), Contract_Month-to-month (11.4%), MonthlyCharges (7.9%), OnlineSecurity_No (6.1%)

**Serving latency (typical):** p50 < 50ms, p95 < 100ms, p99 < 150ms; SLO targets and error budgets in [docs/SLO.md](docs/SLO.md)

---

## Operations

The full operational reference lives in **[docs/OPERATIONS.md](docs/OPERATIONS.md)**: configuration and environment variables, the monitoring stack, drift detection internals, MLflow registry workflows, Kubernetes deployment, security and secrets management, load testing, the complete command reference, and troubleshooting (including the harmless multiprocessing warnings during training).

**Production hardening checklist (abridged):** mandatory auth + TLS, secrets manager, cloud model storage (S3/GCS), remote MLflow backend, centralised logging and alerting, model promotion approvals, A/B rollout and rollback procedures. The full checklist with rationale is in [docs/OPERATIONS.md](docs/OPERATIONS.md).

---

## FAQ

**Why Docker-only execution?** Reproducibility (anyone can run it with `make setup`), production parity (train where you deploy), and zero dependency debugging for reviewers.

**How did you validate model performance?** 3-way split: hyperparameters and threshold selected on the validation set; the held-out test set evaluated once at the tuned threshold for the headline numbers. Validation-set numbers appear only for labelled comparison.

**Why an F1-optimised threshold instead of 0.5?** The default assumes balanced classes and symmetric costs, wrong on both counts here (~26.5% churn rate; false negatives cost 10-50x more). For production I'd move to cost-sensitive thresholds once actual business costs are quantified.

---

## Licence

MIT. See the repository [LICENSE](../LICENSE). The bundled Telco Customer Churn dataset is IBM sample data (via Kaggle) and carries its own terms.

**Dataset source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) | **Part of:** [ML Portfolio](../)
