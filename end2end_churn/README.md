# Customer Churn Prediction - Production ML Pipeline

[![CI Pipeline](https://github.com/Medesen/portfolio/workflows/CI%20-%20End2End%20Churn%20Prediction/badge.svg)](https://github.com/Medesen/portfolio/actions/workflows/ci-end2end-churn.yml)

A production-ready ML service that predicts customer churn for a telecom dataset. I built this to demonstrate end-to-end ML engineering: from training with hyperparameter tuning to serving via REST API, with MLflow experiment tracking, drift detection, monitoring dashboards, and a full CI/CD pipeline. Everything runs in Docker with zero local setup required.

## Summary

**Domain:** Telecom customer churn prediction  
**Dataset:** Telco Customer Churn (7,043 customers, 20 features)  
**Task:** Binary classification  
**Best Performance:** ROC AUC 0.836 (validation set)  
**Supported Models:** Random Forest, XGBoost, Logistic Regression  
**Tech Stack:** FastAPI, MLflow, Prometheus/Grafana, Docker, Kubernetes  
**Test Coverage:** 175+ tests, 89% coverage

**Model Performance:**
- ROC AUC: 0.836
- Precision (Churn): 66%
- Recall (Churn): 46%
- F1-optimized threshold: ~0.35 (vs default 0.5)

**Business Context:** In churn prediction, false negatives (missed churners) cost significantly more than false positives (unnecessary retention campaigns) due to lost customer lifetime value. Industry estimates suggest this cost ratio is often 20x or higher. The model uses threshold tuning to balance these asymmetric costs, prioritizing recall while maintaining acceptable precision.

---

## Quick Start (5 Minutes)

### Prerequisites

- Docker Desktop installed ([Get Docker](https://docs.docker.com/get-docker/))
  - Includes Docker Compose (no separate install needed)
  - Works on Linux, macOS, and Windows
  - **Note:** Requires Docker Compose V2 (`docker compose` command). If you have an older Docker installation that only supports V1 (`docker-compose` with hyphen), you'll need to either upgrade Docker or manually replace `docker compose` with `docker-compose` in the setup script.
- CPU: 2 cores minimum
- RAM: 4 GB minimum
- Disk space: ~2 GB

### Docker-Only Execution

This project runs exclusively via Docker using `make` commands. Manual Python setup is not supported.

**Why Docker-only?**
- Zero dependency issues for reviewers and hiring managers
- Production parity - train in the same environment where models deploy
- Reproducibility - identical execution across all platforms
- No virtual environment management required

Just install Docker and run `make setup`. No Python, pip, or conda needed.

### Platform-Specific Notes

**Linux & macOS:** All commands work as shown. GNU Make is pre-installed.

**Windows:** This README uses `make` commands (e.g., `make up`, `make test`) which are NOT available by default on Windows. You have three options:

1. **Install Make** (recommended for best experience):
   ```powershell
   choco install make
   ```
   After installation, all `make` commands in this README work as shown.

2. **Use PowerShell script for setup, then Docker commands**:
   - Use `.\setup.ps1` for initial setup (works out of the box, no tools needed)
   - For all other commands, use [Direct Docker Commands](#direct-docker-commands) shown later in this README
   - Example: Instead of `make up`, use `docker compose up -d api`

3. **Use Git Bash or WSL2** (includes Make pre-installed):
   - Use `make setup` for setup (just like Linux/macOS)
   - All `make` commands work as shown
   - Full compatibility with all commands

### One-Command Setup

**For Linux/macOS:**
```bash
# Clone the repository and navigate to project
git clone https://github.com/Medesen/portfolio.git
cd portfolio/end2end_churn

# Run automated setup (builds containers, trains model)
make setup
```

**For Windows:**

Option 1 - PowerShell (Recommended):
```powershell
# Clone the repository and navigate to project
git clone https://github.com/Medesen/portfolio.git
cd portfolio\end2end_churn

# Run automated setup (builds containers, trains model)
.\setup.ps1
```

Option 2 - Git Bash / WSL2:
```bash
# Clone the repository and navigate to project
git clone https://github.com/Medesen/portfolio.git
cd portfolio/end2end_churn

# Run automated setup (builds containers, trains model)
make setup
```

**Setup process:**
1. Builds Docker containers (~2-3 min)
2. Trains initial model (Random Forest, ~1-2 min)

### Try It Out

After setup completes, you can immediately start using the service:

```bash
# Start API service
make up

# Test prediction
make test-api

# Stop service
make down

# See all commands
make help
```

Run `make help` to see all available commands.

---

## What This Project Demonstrates

### Technical Skills

I built a complete ML service with training, serving, and monitoring. The training pipeline handles three algorithms (Random Forest, XGBoost, Logistic Regression) with automated hyperparameter tuning using grid search and cross-validation. Every training run logs to MLflow for experiment tracking, and the model registry manages version promotion through staging to production.

The API layer uses FastAPI with Pydantic for request validation, includes rate limiting per endpoint, and implements proper error handling with request tracing. I added drift detection that monitors numeric features (mean/std changes), categorical features (Population Stability Index), and prediction distributions. When drift crosses thresholds, the system can trigger automatic retraining.

The monitoring stack uses Prometheus for metrics collection (latency histograms, request counts, drift events) and Grafana for visualization. I built three dashboards: API overview, latency/SLO tracking, and ML-specific metrics. The test suite has 175+ tests covering unit, integration, and end-to-end workflows, achieving 89% coverage.

Deployment is fully containerized—everything runs in Docker with no local Python setup required. The CI/CD pipeline on GitHub Actions runs linting, tests, security scanning, and load testing on every push. Kubernetes manifests include auto-scaling, health probes, resource limits, and ingress configuration.

### Key Technical Decisions

**Docker-only execution:** Rather than support both local and containerized workflows, I made Docker mandatory. This eliminates "works on my machine" issues and ensures hiring managers can run the project with just `make setup`. It also enforces production parity—training happens in the same environment where models deploy.

**Threshold optimization:** The default 0.5 classification threshold ignores both class imbalance and business costs. For churn prediction, missing a churner (false negative) costs significantly more than an unnecessary retention campaign (false positive)—often 20x or more due to lost lifetime value. I implemented F1-optimized thresholds (~0.35 for this dataset) and included alternative strategies like precision-constrained and cost-sensitive thresholds.

**MLflow tracking as default:** I made experiment tracking mandatory rather than optional. This prevents lost experiments and adds minimal overhead (~100ms per run) while ensuring complete reproducibility. The model registry supports proper staging workflows, though local file loading remains the default for simplicity.

**Drift detection trade-offs:** I implemented statistical drift detection (mean/std for numeric, PSI for categorical) rather than using model-based approaches. This is faster and interpretable but may miss subtle drifts that don't affect distributions. The thresholds are configurable since optimal values depend on business tolerance for false alarms vs missed drift.

---

## Documentation

- **[README.md](README.md)** (this file) - Quick start and overview
- **[Makefile](Makefile)** - Command shortcuts (run `make help`)
- **Project structure** - See [Project Structure](#project-structure) section below
- **Development history** - See commit history for iteration details

---

## Testing

### Test Suite Overview

The project includes 175+ tests (89% coverage) demonstrating patterns for all major components: preprocessing, training pipeline, model factory, API endpoints, drift detection, threshold tuning, and monitoring integration.

**Test Results:** 175+ tests passing | 89% code coverage

**Production considerations:** This test suite is comprehensive for a portfolio project but not exhaustive. For production, I would add property-based testing, chaos engineering tests, performance regression tests, and more extensive edge case coverage.

### Running Tests

All tests run inside Docker to ensure consistency with the deployment environment:

```bash
# Run all tests (displays test results and pass/fail status)
make test

# Run tests with coverage report (generates detailed coverage breakdown)
make test-coverage

# Run by category
make test-unit              # Unit tests (fast)
make test-integration       # API integration tests
make test-e2e              # End-to-end workflows

# Run specific features
make test-schema           # Schema alignment tests
make test-drift           # Drift detection tests
make test-api             # API endpoint tests
```

**Why Docker-only testing?** Running tests through Docker ensures consistent environment, all dependencies are present, same behavior as production deployment, and no need to set up local Python environment.

---

## Dataset

**Domain:** Telecom customer churn  
**Source:** Kaggle Telco Customer Churn (IBM Sample Data)  
**Size:** 7,043 customers, 20 features

**Feature Categories:**
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Account: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

**Target:** Churn (Yes/No) - Binary classification

**Class Distribution:** ~26.5% churn rate (imbalanced dataset)

---

## API Endpoints

Available at http://localhost:8000:

- **GET /** - API information
- **GET /health** - Detailed health check
- **GET /healthz** - Liveness probe (Kubernetes)
- **GET /readyz** - Readiness probe (Kubernetes)
- **POST /predict** - Churn prediction
- **POST /drift** - Analyze data drift
- **GET /drift/info** - Drift configuration
- **POST /retrain** - Trigger retraining
- **GET /metrics** - Prometheus metrics
- **GET /docs** - Swagger UI

### Making Predictions

**Using curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

**Using Python:**
```python
import requests

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No"
}

response = requests.post("http://localhost:8000/predict", json=customer)
print(response.json())
# Output: {"churn_probability": 0.56, "churn_prediction": "Yes", "risk_level": "Medium", ...}
```

**Interactive Documentation:**
1. Start service: `make up`
2. Open http://localhost:8000/docs
3. Try the API interactively

---

## Model Training

All training runs via Docker:

```bash
# Standard training (Random Forest)
make train

# Quick training (fast iteration)
make train-quick

# Production training (extensive search)
make train-prod

# Train specific models
make train-rf          # Random Forest
make train-xgboost     # XGBoost
make train-logreg      # Logistic Regression

# Compare all models
make compare-models
make compare-models-quick

# Train and register in MLflow
make train-register

# View experiments
make mlflow-ui         # Opens http://localhost:5000
```

**Training process:**
1. Loads data with preprocessing (scaling, one-hot encoding)
2. Performs 3-way split (train/validation/test)
3. Runs grid search with cross-validation
4. Optimizes classification threshold for imbalanced data
5. Logs everything to MLflow (parameters, metrics, artifacts)
6. Saves model with metadata and checksums

---

## Configuration

### Environment Variables

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

### Config Profiles

Training configurations in `config/`:

- `train_config.yaml` - Default (balanced)
- `train_config_quick.yaml` - Fast iteration
- `prod.yaml` - Production (extensive search)

---

## Monitoring Stack

Start full monitoring (Prometheus + Grafana):

```bash
# Start all services
make up-monitoring

# Access dashboards
make grafana-ui           # http://localhost:3000 (admin/admin)
make prometheus-ui        # http://localhost:9090

# Check metrics
make metrics             # View raw Prometheus metrics
make monitoring-status   # Health check all services

# Stop monitoring
make down-monitoring
```

**Grafana Dashboards:**
- Churn API Overview (latency, throughput, errors)
- Latency & SLO Monitoring (p50, p95, p99)
- ML Metrics (drift, predictions, model age)

**Prometheus Metrics:**
- `churn_prediction_requests_total` - Request counts
- `churn_prediction_request_duration_seconds` - Latency histogram
- `churn_predictions_total` - Predictions by class
- `churn_drift_detected_total` - Drift detection events
- `churn_model_age_days` - Model age

---

## Drift Detection

Monitor data distribution changes:

```bash
# Test drift detection
make test-drift

# Check configuration
make drift-info

# Trigger retraining
make retrain
```

**Drift Types:**
- **Numeric**: Relative change in mean/std (threshold: 20%)
- **Categorical**: Population Stability Index (threshold: 0.25)
- **Prediction**: Change in prediction rate (threshold: 10%)

**PSI Thresholds:**
- < 0.1: No significant change
- 0.1 - 0.25: Moderate change (monitor)
- ≥ 0.25: Significant change (retrain)

**Configuration:**
```yaml
# docker-compose.yml
environment:
  - DRIFT_THRESHOLD_NUMERIC=0.2
  - DRIFT_THRESHOLD_CATEGORICAL=0.25
  - DRIFT_THRESHOLD_PREDICTION=0.1
  - AUTO_RETRAIN_ON_DRIFT=false
  - MIN_RETRAIN_INTERVAL_HOURS=24
```

---

## MLflow Model Registry

Manage model lifecycle:

```bash
# Register model
make train-register

# Promote to staging
make registry-promote VERSION=3 STAGE=Staging

# Promote to production
make registry-promote VERSION=3 STAGE=Production

# List models and versions
make registry-list
make registry-versions

# Get model info
make registry-info STAGE=Production
make registry-info VERSION=3

# Load from registry
docker compose run --rm \
  -e MODEL_SOURCE=registry \
  -e MODEL_STAGE=Production \
  -p 8000:8000 api
```

**Model Stages:** None → Staging → Production → Archived

---

## Kubernetes Deployment

Production deployment manifests in `k8s/`:

```bash
# Deploy to cluster
kubectl apply -f k8s/

# Check status
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -l app=churn-prediction

# Scale manually
kubectl scale deployment churn-prediction --replicas=5
```

**Features:**
- 3 replicas with auto-scaling (3-10 pods)
- Resource limits (CPU: 250m-1000m, Memory: 512Mi-1Gi)
- Health probes (liveness, readiness, startup)
- ConfigMap for configuration
- Ingress with TLS and rate limiting
- PersistentVolumeClaim for models

See `k8s/README.md` for detailed deployment guide.

---

## Security

### Secrets Management

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

### Container Security Scanning

```bash
# Scan for vulnerabilities
make security-scan

# Critical/high only
make security-scan-critical
```

Results uploaded to GitHub Security tab in CI/CD.

### Authentication

Optional bearer token authentication:

```bash
# Test with auth
export SERVICE_TOKEN="secret-token"
curl -H "Authorization: Bearer secret-token" \
  http://localhost:8000/predict -d @test_request.json
```

---

## Performance

### Load Testing

```bash
# Start service
make up

# Run load test (50 users, 60s)
make load-test

# Interactive UI
make load-test-ui         # http://localhost:8089
```

**SLO Targets (CI):**
- p95 latency < 500ms
- p99 latency < 1000ms
- Error rate < 1%

**Production SLOs:**
- p95 < 200ms
- p99 < 500ms
- Error rate < 0.1%

### Rate Limiting

Configured per endpoint:
- `/predict`: 100 requests/minute
- `/drift`: 20 requests/minute

Returns 429 (Too Many Requests) when exceeded.

---

## CI/CD Pipeline

GitHub Actions workflow runs on every PR/push:

1. **Code Quality** - flake8, black, isort, mypy
2. **Unit Tests** - 175+ tests with coverage
3. **Integration Tests** - API endpoint validation
4. **E2E Tests** - Full training workflows
5. **Docker Build** - Production image validation
6. **Security Scan** - Trivy vulnerability scanning
7. **Load Testing** - SLO validation
8. **Test Summary** - Aggregate results

View results: [GitHub Actions](../../actions/workflows/ci-end2end-churn.yml)

---

## Project Structure

```
end2end_churn/
├── src/                          # Source code
│   ├── api/                      # FastAPI service
│   ├── data/                     # Data loading & preprocessing
│   ├── models/                   # Model pipelines & factory
│   ├── training/                 # Training & tuning
│   ├── evaluation/               # Metrics & visualizations
│   ├── config.py                 # Configuration management
│   └── utils/                    # Logging, I/O, metrics, drift
├── data/                         # Training data
├── models/                       # Saved models
├── diagnostics/                  # Plots and reports
├── logs/                         # Application logs
├── mlruns/                       # MLflow tracking
├── config/                       # Config files
├── k8s/                         # Kubernetes manifests
├── grafana/                      # Grafana dashboards
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── train.py                      # Training orchestration
├── serve.py                      # FastAPI service
├── requirements.txt              # Dependencies
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Orchestration
├── Makefile                      # Command shortcuts
└── README.md                     # This file
```

---

## All Available Commands

### Quick Reference

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
make test-e2e          # End-to-end tests
make test-coverage     # With coverage
make test-api          # Test prediction endpoint
make test-drift        # Test drift detection
make load-test         # Performance testing

# Monitoring
make logs              # View logs
make health            # Check health
make metrics           # View Prometheus metrics
make grafana-ui        # Open Grafana
make prometheus-ui     # Open Prometheus
make mlflow-ui         # Open MLflow
make docs              # Open API docs

# Model Registry
make registry-list              # List registered models
make registry-versions          # List model versions
make registry-promote           # Promote model stage
make registry-info             # Get model details

# Security
make security-scan              # Full security scan
make security-scan-critical     # Critical/high only

# Utilities
make shell             # Open bash in container
make status            # Project status
make clean             # Remove containers
make clean-all         # Complete cleanup
```

### Direct Docker Commands

For Windows users without Make, or if you prefer Docker commands directly:

```bash
# Core operations (these are what the Makefile runs internally)
docker compose build                                    # Build image
docker compose run --rm api python train.py            # Train model
docker compose up -d api                               # Start API
docker compose down                                    # Stop services

# Training variations
docker compose run --rm api python train.py --config config/train_config_quick.yaml  # Quick training
docker compose run --rm api python train.py --compare-all  # Compare all models

# Testing
docker compose run --rm api pytest tests/ -v           # Run all tests
docker compose run --rm api pytest tests/unit/ -v      # Unit tests only

# API testing (requires API to be running)
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_request.json

# Monitoring
docker compose up -d api prometheus grafana  # Start with monitoring
docker compose logs api  # View logs

# MLflow UI (run in separate terminal)
docker compose run --rm -p 5000:5000 --entrypoint mlflow api ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
# Then open http://localhost:5000 in browser

# Cleanup
docker compose down  # Stop services
docker compose down -v  # Stop and remove volumes
```

---

## Key Technologies

**ML/Data:**
- scikit-learn (Random Forest, Logistic Regression)
- XGBoost (gradient boosting)
- pandas, numpy (data processing)
- MLflow (experiment tracking, model registry)

**API/Service:**
- FastAPI (REST API)
- Pydantic (validation)
- Uvicorn (ASGI server)
- slowapi (rate limiting)

**Monitoring:**
- Prometheus (metrics)
- Grafana (dashboards)
- Structured logging (request tracing)

**DevOps:**
- Docker & Docker Compose
- Kubernetes
- GitHub Actions (CI/CD)
- Trivy (security scanning)
- Locust (load testing)

**Testing:**
- pytest (test framework)
- coverage (code coverage)
- pytest-xdist (parallel execution)

---

## Production Considerations

### Security Hardening

Before production deployment:

- [ ] **Enable mandatory authentication** - Set `SERVICE_TOKEN` (currently optional)
- [ ] **Use secrets manager** - Docker secrets or K8s Secrets (supported via `SERVICE_TOKEN_FILE`)
- [ ] **Enable TLS/HTTPS** - At ingress/load balancer
- [ ] **Restrict metrics endpoint** - Internal network only
- [ ] **Disable API docs** - Or require authentication
- [ ] **Network segmentation** - Private subnet with API gateway
- [ ] **Review rate limits** - Tune based on capacity
- [ ] **Validate inputs strictly** - Reject (not warn) on invalid data

### Monitoring & Alerting

- [ ] **Centralized logging** - ELK, Splunk, CloudWatch
- [ ] **Prometheus scraping** - Configure scrape targets
- [ ] **Critical alerts** - Drift, errors, latency, availability
- [ ] **Grafana dashboards** - Customize for your metrics
- [ ] **Distributed tracing** - OpenTelemetry + Jaeger (optional)

### Infrastructure

- [ ] **Cloud storage** - S3/GCS for models (not local files)
- [ ] **Remote MLflow backend** - PostgreSQL + S3 artifacts
- [ ] **Resource tuning** - Based on load testing
- [ ] **Auto-scaling** - Configure HPA thresholds
- [ ] **Backup strategy** - Models, data, MLflow DB

### ML Operations

- [ ] **Model promotion workflow** - Require approval for Production
- [ ] **A/B testing** - Staging vs Production traffic split
- [ ] **Rollback procedures** - Test and document
- [ ] **Scheduled retraining** - Cron/Airflow based on drift
- [ ] **Tune drift thresholds** - Reduce false positives
- [ ] **Model governance** - Document assumptions, lineage

---

## Advanced Features

### Threshold Tuning

The default 0.5 classification threshold is inappropriate for imbalanced data and ignores business costs. **In churn prediction, false negatives (missed churners) typically cost 10-50x more than false positives** (unnecessary retention campaigns) due to lost customer lifetime value.

The model automatically optimizes thresholds using F1 maximization, which provides a reasonable balance for this cost asymmetry. For this dataset, the optimal threshold is ~0.35 instead of 0.5, significantly improving recall while maintaining acceptable precision.

**Available Strategies:**
- **F1 Maximization** (default) - Balances precision/recall for imbalanced data
- **Precision-Constrained** - Maximize recall with minimum precision (e.g., "need ≥70% precision")
- **Top-K Selection** - Flag exactly K highest-risk customers (budget constraints)
- **Cost-Sensitive** - Minimize expected cost when specific business costs are known

All strategies are computed during training and saved in metadata. The API automatically applies the tuned threshold.

**View threshold analysis:** `diagnostics/threshold_analysis_*.png`

### Type Safety

- Strict mypy configuration (`disallow_untyped_defs = true`)
- TypedDict for structured data
- Generic type annotations throughout
- 100% type coverage in core layers

### Request Tracing

Every request gets a unique correlation ID:
- Appears in all logs: `[req-abc-123]`
- Returned in response and `X-Request-ID` header
- Client-provided IDs supported for distributed tracing

```bash
# Find all logs for a request
grep "req-abc-123" logs/churn_service.log
```

### Async Concurrency

CPU-bound operations (inference, drift) offloaded to thread pool using `run_in_threadpool()` - keeps event loop responsive under concurrent load.

### Model Integrity

SHA256 checksums validate model files:
- Computed during training
- Verified during loading
- Detects corruption and tampering
- 16 comprehensive tests (85% I/O coverage)

---

## Performance Metrics

**Model Performance (Validation Set):**
- ROC AUC: ~0.836
- Precision (Churn): ~66%
- Recall (Churn): ~46%
- Accuracy: ~79%

**Business Impact (Illustrative):**
The model catches 46% of churners (172 out of 374) with 66% precision. In typical telecom scenarios, missing a churner costs significantly more than unnecessary retention efforts—often **estimated at 20x or higher** due to lost lifetime value vs. relatively low-cost retention campaigns. 

The model's threshold tuning (F1 maximization) balances these asymmetric costs, prioritizing recall while maintaining acceptable precision. The exact cost ratio varies by business context, customer segment, and retention strategy.

**Top Predictive Features:**
1. tenure (customer age) - 14.3%
2. TotalCharges - 11.5%
3. Contract_Month-to-month - 10.7%
4. MonthlyCharges - 7.7%
5. OnlineSecurity_No - 6.5%

**Latency (typical):**
- p50: < 50ms
- p95: < 100ms  
- p99: < 150ms

---

## Troubleshooting

### Docker Issues

**API fails health check:**
```bash
# Check if model loaded successfully
docker compose logs api
```

**Port already in use:**
```yaml
# Edit docker-compose.yml to change host port
ports:
  - "8001:8000"  # Use different host port
```

**Prometheus shows no targets:**
```bash
# Verify API is healthy
docker compose ps
```

**Cannot connect to the Docker daemon:**
- Start Docker Desktop
- Verify Docker is running: `docker ps`

**Out of disk space:**
```bash
# Clean up old images and containers
docker system prune -a
```

**"Docker Compose is configured to build using Bake, but buildx isn't installed":**
- Harmless warning from Docker Compose V2
- Builds work identically - Docker automatically falls back to standard builder
- To eliminate: Update Docker Desktop to the latest version or install buildx
- Safe to ignore - no impact on functionality

### Application Issues

**Multiprocessing Cleanup Warnings:** During training (`make train` or `make setup`), you may see `ChildProcessError: [Errno 10] No child processes` warnings after the model finishes training. These are **harmless** multiprocessing cleanup warnings and do not affect model quality.

**What's happening:** The training process uses parallel processing (`GridSearchCV` with `n_jobs=-1`) to speed up hyperparameter search by testing multiple parameter combinations simultaneously. When the Docker container shuts down after training completes, Python's ResourceTracker attempts to clean up worker processes that have already terminated, resulting in these warning messages.

**Why this occurs in Docker:** Short-lived containers exit quickly after the main process completes, creating a race condition where child processes terminate before Python's cleanup code runs. This is a known behavior in scikit-learn's parallel processing (see [Python issue #38119](https://bugs.python.org/issue38119)).

**Verification:** Check for the "Training workflow complete!" message and verify model files were created:
```bash
ls models/  # Should show churn_model_latest.joblib
```

**To eliminate warnings (optional):** Set `n_jobs: 1` in your training config. This disables parallel processing, making training slower but removing the warnings.

**Model not found:**
```bash
# Train model first
make train

# Verify files exist
ls models/
```

**Tests fail:**
```bash
# Use the correct test command
make test

# Check Docker is running
docker ps
```

### More Help

See detailed information in:
- [Makefile](Makefile) - Run `make help` for all commands
- [GitHub Actions](.github/workflows/ci-end2end-churn.yml) - CI/CD pipeline details

---

## Frequently Asked Questions

### Why Docker-only execution?

Three reasons:
1. **Reproducibility** - Anyone can run it with just `make setup`
2. **Production parity** - Train in the same environment where models deploy
3. **Zero dependency issues** - No virtual environment conflicts

For a portfolio, it's critical that hiring managers can easily run the project without debugging pip conflicts.

### What would you change for production?

Several things:
- Replace local file loading with cloud storage (S3/GCS)
- Use remote MLflow backend (PostgreSQL + S3 artifacts)
- Enable mandatory authentication (currently optional)
- Add comprehensive alerting for drift and errors
- Implement A/B testing for model promotion
- Add distributed tracing (OpenTelemetry)
- More extensive test coverage for edge cases

See [Production Considerations](#production-considerations) section for full checklist.

### How did you validate the model performance?

I used standard classification metrics (ROC AUC, precision, recall, F1) on a held-out test set with 3-way data splitting (train/validation/test). The validation set is used for hyperparameter tuning, and the test set provides final unbiased performance estimates. Cross-validation during grid search provides additional validation.

### Why F1-optimized threshold instead of default 0.5?

The default 0.5 threshold assumes balanced classes and equal costs for false positives vs false negatives. In churn prediction, this is wrong on both counts:
- Classes are imbalanced (~26.5% churn rate)
- False negatives (missed churners) cost 20x more than false positives due to lost lifetime value

F1 optimization provides a reasonable balance. For production, I'd recommend cost-sensitive thresholds once actual business costs are quantified.

### Does this work on Mac/Windows/Linux?

Yes. Docker Desktop works on all three platforms:
- **Linux:** Native Docker support + Make pre-installed
- **macOS:** Docker Desktop includes everything needed + Make pre-installed
- **Windows:** Docker Desktop with WSL2 backend. Make not included by default.

**Windows users:** See [Platform-Specific Notes](#platform-specific-notes) at the top for three setup options. The short version: either install Make via `choco install make`, use the provided PowerShell setup script (`.\setup.ps1`), or use the [Direct Docker Commands](#direct-docker-commands) instead of `make` commands.

### How much does it cost to run?

Zero dollars for local execution. Everything runs on your machine using open-source tools. For production cloud deployment, costs would depend on instance sizes, traffic volume, and cloud provider (AWS, GCP, Azure).

---

## License

This project is part of an ML portfolio. See main repository for license details.

---

## Related Links

- **Dataset Source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Part of ML Portfolio:** [portfolio](../)

---

**Last Updated:** November 2025  
**Docker Support:** Linux, macOS, Windows  
**Total Setup Time:** ~5 minutes
