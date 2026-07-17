# Operations Guide

Deep-dive operational reference for the churn prediction service. The
[README](../README.md) covers the quick start and project overview; this
document holds the full command reference, configuration, monitoring,
deployment, and troubleshooting detail.

## Contents

- [Windows setup options](#windows-setup-options)
- [Direct Docker commands](#direct-docker-commands)
- [Full command reference](#full-command-reference)
- [Configuration](#configuration)
- [Monitoring stack](#monitoring-stack)
- [Drift detection](#drift-detection)
- [MLflow model registry](#mlflow-model-registry)
- [Kubernetes deployment](#kubernetes-deployment)
- [Security](#security)
- [Load testing](#load-testing)
- [Advanced features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## Windows setup options

The README uses `make` commands, which are not available by default on
Windows. Three options:

1. **Install Make** (best experience):
   ```powershell
   choco install make
   ```
   All `make` commands then work as shown.

2. **PowerShell script for setup, Docker commands for the rest**:
   - `.\setup.ps1` for initial setup (no extra tools needed)
   - Then use [Direct Docker commands](#direct-docker-commands), e.g.
     `docker compose up -d api` instead of `make up`.

3. **Git Bash or WSL2** (Make pre-installed): use `make setup` and every
   other `make` command exactly as on Linux/macOS.

---

## Direct Docker commands

For Windows users without Make, or if you prefer Docker commands directly:

```bash
# Core operations (these are what the Makefile runs internally)
docker compose build                                    # Build image
docker compose run --rm api python train.py            # Train model
docker compose up -d api                               # Start API
docker compose down                                    # Stop services

# Training variations
docker compose run --rm api python train.py --config config/train_config_quick.yaml
docker compose run --rm api python scripts/compare_models.py  # Compare all 3 models

# Testing (tests are organized by pytest markers, not directories)
docker compose run --rm api pytest tests/ -v           # Run all tests
docker compose run --rm api pytest -m unit -v          # Unit tests only

# API testing (requires API to be running)
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_request.json

# Monitoring
docker compose up -d api prometheus grafana  # Start with monitoring
docker compose logs api                      # View logs

# MLflow UI (run in separate terminal)
docker compose run --rm -p 5000:5000 --entrypoint mlflow api ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
# Then open http://localhost:5000 in browser

# Cleanup
docker compose down     # Stop services
docker compose down -v  # Stop and remove volumes
```

---

## Full command reference

```bash
# See all available commands
make help

# Setup & Deployment
make setup              # Build + train model
make build              # Build Docker image
make up                 # Start API
make down               # Stop API
make restart            # Restart after updates
make up-monitoring      # Start with Prometheus/Grafana
make down-monitoring    # Stop all services

# Training
make train              # Train model (default config)
make train-quick        # Quick training
make train-prod         # Production training
make train-rf           # Random Forest
make train-xgboost      # XGBoost
make train-logreg       # Logistic Regression
make compare-models     # Compare all models
make train-register     # Train + register in MLflow

# Testing
make test               # All tests
make test-unit          # Unit tests
make test-integration   # Integration tests
make test-e2e           # End-to-end tests
make test-coverage      # With coverage
make test-api           # Test prediction endpoint
make test-drift         # Test drift detection
make load-test          # Performance testing

# Monitoring
make logs               # View logs
make health             # Check health
make metrics            # View Prometheus metrics
make grafana-ui         # Open Grafana
make prometheus-ui      # Open Prometheus
make mlflow-ui          # Open MLflow
make docs               # Open API docs

# Model Registry
make registry-list      # List registered models
make registry-versions  # List model versions
make registry-promote   # Promote model stage
make registry-info      # Get model details

# Security
make security-scan          # Full security scan
make security-scan-critical # Critical/high only

# Utilities
make shell              # Open bash in container
make status             # Project status
make clean              # Remove containers
make clean-all          # Complete cleanup
```

---

## Configuration

### Environment variables

Create `.env` file (see `.env.example`):

```bash
# Service
LOG_LEVEL=INFO
SERVICE_TOKEN_FILE=/run/secrets/service_token  # Recommended
SERVICE_TOKEN=your-token                        # Or direct env var
MAX_BATCH_SIZE=1000
MAX_REQUEST_SIZE_MB=10
REQUEST_TIMEOUT_SECONDS=30

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PREDICT=100/minute
RATE_LIMIT_DRIFT=20/minute

# Drift Detection
DRIFT_THRESHOLD_NUMERIC=0.2
DRIFT_THRESHOLD_CATEGORICAL=0.25
DRIFT_THRESHOLD_PREDICTION=0.1
AUTO_RETRAIN_ON_DRIFT=false
MIN_RETRAIN_INTERVAL_HOURS=24

# Model Registry
MODEL_SOURCE=local              # or 'registry'
MODEL_STAGE=Production
MODEL_NAME=churn_prediction_model

# Monitoring
GRAFANA_PASSWORD=admin
```

### Config profiles

Training configurations in `config/`:

- `train_config.yaml` — Default (balanced)
- `train_config_quick.yaml` — Fast iteration
- `prod.yaml` — Production (extensive search)

---

## Monitoring stack

Start full monitoring (Prometheus + Grafana):

```bash
make up-monitoring       # Start all services
make grafana-ui          # http://localhost:3000 (admin/admin)
make prometheus-ui       # http://localhost:9090
make metrics             # View raw Prometheus metrics
make monitoring-status   # Health check all services
make down-monitoring     # Stop monitoring
```

**Grafana dashboards:**
- Churn API Overview (latency, throughput, errors)
- Latency & SLO Monitoring (p50, p95, p99) — targets in [SLO.md](SLO.md)
- ML Metrics (drift, predictions, model age)

**Prometheus metrics:**
- `churn_prediction_requests_total` — Request counts
- `churn_prediction_request_duration_seconds` — Latency histogram
- `churn_predictions_total` — Predictions by class
- `churn_drift_detected_total` — Drift detection events
- `churn_model_age_days` — Model age

---

## Drift detection

Monitor data distribution changes:

```bash
make test-drift    # Test drift detection
make drift-info    # Check configuration
make retrain       # Trigger retraining
```

**Drift types:**
- **Numeric**: Relative change in mean/std (threshold: 20%), plus a KS test
  against a stored reference sample
- **Categorical**: Population Stability Index (threshold: 0.25)
- **Prediction**: Change in predicted-positive rate at the deployed threshold
  (threshold: 10%)

**PSI thresholds:**
- < 0.1: No significant change
- 0.1 – 0.25: Moderate change (monitor)
- ≥ 0.25: Significant change (retrain)

**Configuration** (docker-compose.yml):
```yaml
environment:
  - DRIFT_THRESHOLD_NUMERIC=0.2
  - DRIFT_THRESHOLD_CATEGORICAL=0.25
  - DRIFT_THRESHOLD_PREDICTION=0.1
  - AUTO_RETRAIN_ON_DRIFT=false
  - MIN_RETRAIN_INTERVAL_HOURS=24
```

Retraining is rate-limited by an atomic file-lock reservation
(`src/api/retraining.py`): concurrent triggers cannot launch duplicate runs,
and a failed run still consumes the interval (crash-loop protection). A newly
trained model is saved but never auto-deployed — deploy with `make restart`
after reviewing `diagnostics/`.

---

## MLflow model registry

Manage model lifecycle:

```bash
make train-register                              # Register model
make registry-promote VERSION=3 STAGE=Staging    # Promote to staging
make registry-promote VERSION=3 STAGE=Production # Promote to production
make registry-list                               # List models
make registry-versions                           # List versions
make registry-info STAGE=Production              # Get model info

# Serve from the registry instead of local files
docker compose run --rm \
  -e MODEL_SOURCE=registry \
  -e MODEL_STAGE=Production \
  -p 8000:8000 api
```

**Model stages:** None → Staging → Production → Archived

---

## Kubernetes deployment

Production deployment manifests in `k8s/`:

```bash
kubectl apply -f k8s/                       # Deploy to cluster
kubectl get deployments pods services       # Check status
kubectl logs -l app=churn-prediction        # View logs
kubectl scale deployment churn-prediction --replicas=5
```

**Features:**
- 3 replicas with auto-scaling (3–10 pods)
- Resource limits (CPU: 250m–1000m, Memory: 512Mi–2Gi)
- Health probes (liveness, readiness, startup)
- ConfigMap for configuration
- Ingress with TLS and rate limiting
- PersistentVolumeClaim for models
- Non-root container execution

See `k8s/README.md` for the detailed deployment guide.

---

## Security

### Secrets management

**File-based secrets (recommended for production):**
```bash
# Docker
echo "your-token" > secrets/service_token.txt
export SERVICE_TOKEN_FILE=/run/secrets/service_token

# Kubernetes
kubectl create secret generic churn-secrets \
  --from-literal=service-token=YOUR_TOKEN
```

**Environment variable (development only):**
```bash
export SERVICE_TOKEN="dev-token"
```

### Authentication

Optional bearer token authentication (constant-time comparison, tested
including the 401 paths):

```bash
export SERVICE_TOKEN="secret-token"
curl -H "Authorization: Bearer secret-token" \
  http://localhost:8000/predict -d @test_request.json
```

### Container security scanning

```bash
make security-scan           # Scan for vulnerabilities
make security-scan-critical  # Critical/high only
```

In CI the Trivy scan runs as an **advisory** job: findings are reported to the
logs and the GitHub Security tab but do not fail the workflow (new base-image
CVEs shouldn't break unrelated changes). The workflow comments document how to
tighten this to a blocking severity policy with a `.trivyignore` allowlist.

---

## Load testing

```bash
make up            # Start service
make load-test     # Run load test (50 users, 60s)
make load-test-ui  # Interactive UI at http://localhost:8089
```

**SLO targets (CI):** p95 < 500ms, p99 < 1000ms, error rate < 1% — the CI job
parses Locust output and fails on violations. Production targets and error
budgets are documented in [SLO.md](SLO.md).

**Rate limiting:** `/predict` 100 req/min, `/drift` 20 req/min per client IP;
429 when exceeded.

---

## Advanced features

### Threshold tuning

The default 0.5 classification threshold ignores both class imbalance and
business costs. Training computes four strategies and saves them all in the
model metadata; the API automatically applies the chosen (F1-maximizing)
threshold:

- **F1 Maximization** (default) — balances precision/recall for imbalanced data
- **Precision-Constrained** — maximize recall with a minimum precision
- **Top-K Selection** — flag exactly K highest-risk customers (budget constraints)
- **Cost-Sensitive** — minimize expected cost when business costs are known

View the threshold analysis plots at `diagnostics/threshold_analysis_*.png`.

### Request tracing

Every request gets a unique correlation ID that appears in all logs
(`[req-abc-123]`), is returned in the response and `X-Request-ID` header, and
honors client-provided IDs for distributed tracing:

```bash
grep "req-abc-123" logs/churn_service.log
```

### Async concurrency

CPU-bound operations (inference, drift analysis) are offloaded to a thread
pool via `run_in_threadpool()`, keeping the event loop responsive under
concurrent load. Request body size is enforced by a pure-ASGI middleware that
counts streamed bytes, so chunked uploads without a Content-Length header
cannot bypass the limit.

### Model integrity

SHA256 checksums validate model files:
- Computed during training (a `.sha256` sidecar next to each model)
- Verified during loading — the service **fails closed**: a checksum mismatch
  or missing sidecar refuses to deserialize the model.
  `ALLOW_UNVERIFIED_MODELS=true` is an explicit, logged dev override.

---

## Troubleshooting

### Docker issues

**API fails health check:**
```bash
docker compose logs api   # Check if model loaded successfully
```

**Port already in use:** edit `docker-compose.yml` to map a different host
port (e.g. `"8001:8000"`).

**Prometheus shows no targets:** `docker compose ps` — verify the API is
healthy.

**Cannot connect to the Docker daemon:** start Docker Desktop; verify with
`docker ps`.

**Out of disk space:** `docker system prune -a`.

**"Docker Compose is configured to build using Bake, but buildx isn't
installed":** harmless Compose V2 warning; builds fall back to the standard
builder. Update Docker Desktop to eliminate it.

### Application issues

**Multiprocessing cleanup warnings after training** (`ChildProcessError:
[Errno 10] No child processes`): harmless. GridSearchCV runs with
`n_jobs=-1`; when the short-lived container exits, Python's ResourceTracker
tries to clean up workers that already terminated (see
[Python issue #38119](https://bugs.python.org/issue38119)). Verify training
succeeded by the "Training workflow complete!" message and
`ls models/` showing `churn_model_latest.joblib`. Set `n_jobs: 1` in the
training config to eliminate the warnings at the cost of slower training.

**Model not found:** `make train`, then verify with `ls models/`.

**Tests fail:** use `make test` (Docker); check Docker is running with
`docker ps`.
