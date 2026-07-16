# Makefile for Customer Churn Prediction Pipeline
# Docker-based ML service with FastAPI

# PHONY targets are not actual files - they're just command names
.PHONY: help setup build train train-quick train-prod train-register \
        train-rf train-xgboost train-logreg compare-models compare-models-quick \
        up up-monitoring down down-monitoring restart logs shell status \
        health test-api test-liveness test-readiness test-auth test-validation docs metrics \
        test-drift test-schema drift-info retrain \
        mlflow-ui monitoring-status prometheus-ui grafana-ui \
        registry-list registry-versions registry-promote registry-info \
        test test-unit test-integration test-e2e test-quick test-coverage \
        security-scan security-scan-critical \
        load-test load-test-ui \
        clean clean-data clean-models clean-all purge \
        quick-update rebuild

# Default target - show help when running 'make' with no arguments
help:
	@echo "=========================================="
	@echo "Churn Prediction - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup & Deployment:"
	@echo "  make setup          - Initial setup (build image, train model)"
	@echo "  make build          - Build Docker image"
	@echo "  make up             - Start API service only"
	@echo "  make up-monitoring  - Start API + Prometheus + Grafana (full stack)"
	@echo "  make down           - Stop API service"
	@echo "  make down-monitoring - Stop all services (API + monitoring)"
	@echo "  make restart        - Restart API (e.g., after model update)"
	@echo ""
	@echo "Single Model Training:"
	@echo "  make train          - Train single model (Random Forest by default)"
	@echo "  make train-quick    - Train single model with quick config (fast)"
	@echo "  make train-prod     - Train single model with production config (thorough)"
	@echo "  make train-register - Train single model and register to MLflow Registry"
	@echo "  make train-rf       - Train Random Forest specifically"
	@echo "  make train-xgboost  - Train XGBoost specifically"
	@echo "  make train-logreg   - Train Logistic Regression specifically"
	@echo ""
	@echo "Multi-Model Comparison:"
	@echo "  make compare-models       - Train ALL 3 models and compare (RF, XGBoost, LogReg)"
	@echo "  make compare-models-quick - Compare all 3 models with quick config (fast)"
	@echo "  make mlflow-ui            - View training experiments in browser"
	@echo ""
	@echo "Model Registry:"
	@echo "  make registry-list     - List all registered models"
	@echo "  make registry-versions - List all model versions"
	@echo "  make registry-promote  - Promote model to stage (requires VERSION and STAGE)"
	@echo "  make registry-info     - Get model info (requires VERSION or STAGE)"
	@echo ""
	@echo "API Operations:"
	@echo "  make health         - Check API health (detailed status)"
	@echo "  make test-liveness  - Check liveness probe (/healthz)"
	@echo "  make test-readiness - Check readiness probe (/readyz)"
	@echo "  make test-api       - Test prediction endpoint with example data"
	@echo "  make test-auth      - Test with authentication (requires SERVICE_TOKEN)"
	@echo "  make test-validation - Test error handling (batch size, invalid data)"
	@echo "  make test-drift     - Test drift detection with multiple scenarios"
	@echo "  make drift-info     - Show drift detection configuration"
	@echo "  make retrain        - Trigger model retraining via API"
	@echo "  make metrics        - View Prometheus metrics"
	@echo "  make docs           - Open API documentation in browser"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration/API tests only"
	@echo "  make test-e2e       - Run end-to-end tests only"
	@echo "  make test-quick     - Run fast tests only (skip slow tests)"
	@echo "  make test-coverage  - Run tests with detailed coverage report"
	@echo ""
	@echo "Monitoring:"
	@echo "  make up-monitoring     - Start full monitoring stack (Prometheus + Grafana)"
	@echo "  make monitoring-status - Check monitoring services status"
	@echo "  make prometheus-ui     - Open Prometheus in browser"
	@echo "  make grafana-ui        - Open Grafana in browser"
	@echo "  make logs              - Show container logs"
	@echo "  make shell             - Open shell in container"
	@echo "  make status            - Show project status"
	@echo ""
	@echo "Security:"
	@echo "  make security-scan          - Scan Docker image for vulnerabilities (all severities)"
	@echo "  make security-scan-critical - Scan for CRITICAL and HIGH vulnerabilities only"
	@echo ""
	@echo "Performance Testing:"
	@echo "  make load-test    - Run headless load test (50 users, 60s, validate SLOs)"
	@echo "  make load-test-ui - Run load test with web UI (http://localhost:8089)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Stop service and remove containers"
	@echo "  make clean-data     - Remove generated diagnostics"
	@echo "  make clean-models   - Remove trained models"
	@echo "  make clean-all      - Complete cleanup (Docker + generated files)"
	@echo "  make purge          - NUCLEAR: Remove ALL generated files (uses Docker for cleanup)"
	@echo ""
	@echo "Development Shortcuts:"
	@echo "  make quick-update   - Retrain model and restart the API"
	@echo "  make rebuild        - Rebuild image, retrain, and start the API"
	@echo ""
	@echo "Examples:"
	@echo "  make setup                                      # First-time setup"
	@echo "  make train                                      # Train single model (Random Forest)"
	@echo "  make compare-models                             # Train all 3 models and compare"
	@echo "  make up                                         # Start API"
	@echo "  make test-api                                   # Test prediction"
	@echo "  make train-logreg                               # Train Logistic Regression"
	@echo "  make mlflow-ui                                  # View all experiments"
	@echo "  make up-monitoring                              # Start full stack (API + monitoring)"
	@echo ""

# =============================================================================
# Setup & Deployment
# =============================================================================

setup: build
	@echo "Running initial setup..."
	@echo ""
	@echo "Creating secrets directory (for optional authentication)..."
	@mkdir -p secrets
	@touch secrets/service_token.txt
	@echo "✓ Secrets directory created"
	@echo ""
	@echo "Training initial model..."
	@$(MAKE) train
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  make up      # Start API"
	@echo "  make test-api # Test prediction"
	@echo ""

build:
	@echo "Building Docker image..."
	@docker compose build
	@echo "✓ Image built"

# =============================================================================
# Model Training
# =============================================================================

train:
	@echo "Training single model with default config (Random Forest)..."
	@docker compose run --rm api python train.py
	@echo "✓ Model trained and logged to MLflow"
	@echo ""
	@echo "View experiment results:"
	@echo "  make mlflow-ui"
	@echo ""

train-quick:
	@echo "Training single model with quick config (fast hyperparameter search)..."
	@docker compose run --rm api python train.py --config config/train_config_quick.yaml
	@echo "✓ Model trained with quick config"
	@echo ""
	@echo "View experiment results:"
	@echo "  make mlflow-ui"
	@echo ""

train-prod:
	@echo "Training single model with production config (thorough hyperparameter search)..."
	@docker compose run --rm api python train.py --config config/prod.yaml
	@echo "✓ Model trained with production config"
	@echo ""
	@echo "View experiment results:"
	@echo "  make mlflow-ui"
	@echo ""

train-register:
	@echo "Training model and registering to MLflow Registry..."
	@docker compose run --rm -e MLFLOW_REGISTER_MODEL=true api python train.py
	@echo "✓ Model trained and registered"
	@echo ""
	@echo "Next steps:"
	@echo "  make registry-list              # View registered models"
	@echo "  make registry-promote VERSION=1 STAGE=Production  # Promote to production"
	@echo ""

train-rf:
	@echo "Training Random Forest model specifically..."
	@docker compose run --rm api python train.py --model-type random_forest
	@echo "✓ Random Forest model trained"
	@echo ""

train-xgboost:
	@echo "Training XGBoost model specifically..."
	@docker compose run --rm api python train.py --model-type xgboost
	@echo "✓ XGBoost model trained"
	@echo ""

train-logreg:
	@echo "Training Logistic Regression model specifically..."
	@docker compose run --rm api python train.py --model-type logistic_regression
	@echo "✓ Logistic Regression model trained"
	@echo ""

# =============================================================================
# Model Comparison
# =============================================================================

compare-models:
	@echo "=========================================="
	@echo "Training ALL 3 MODELS for comparison"
	@echo "=========================================="
	@echo "Models: Random Forest, XGBoost, Logistic Regression"
	@echo "This will take several minutes..."
	@echo ""
	@docker compose run --rm api python scripts/compare_models.py
	@echo ""
	@echo "✓ All 3 models trained and compared!"
	@echo ""
	@echo "Results saved:"
	@echo "  - diagnostics/model_comparison_report_<run_id>.txt"
	@echo "  - diagnostics/model_comparison_YYYYMMDD_HHMMSS.png"
	@echo ""
	@echo "View all training experiments:"
	@echo "  make mlflow-ui"
	@echo ""

compare-models-quick:
	@echo "=========================================="
	@echo "Quick comparison: ALL 3 MODELS"
	@echo "=========================================="
	@echo "Models: Random Forest, XGBoost, Logistic Regression"
	@echo "Using minimal hyperparameters for speed..."
	@echo ""
	@docker compose run --rm api python scripts/compare_models.py --quick
	@echo ""
	@echo "✓ All 3 models compared (quick mode)!"
	@echo ""
	@echo "Results saved:"
	@echo "  - diagnostics/model_comparison_report_<run_id>.txt"
	@echo "  - diagnostics/model_comparison_YYYYMMDD_HHMMSS.png"
	@echo ""

# =============================================================================
# Model Registry
# =============================================================================

registry-list:
	@echo "Listing registered models..."
	@docker compose run --rm api python scripts/manage_registry.py list

registry-versions:
	@echo "Listing model versions..."
	@docker compose run --rm api python scripts/manage_registry.py versions

registry-promote:
	@if [ -z "$(VERSION)" ] || [ -z "$(STAGE)" ]; then \
		echo "Error: VERSION and STAGE required"; \
		echo "Usage: make registry-promote VERSION=1 STAGE=Production"; \
		echo "Valid stages: Staging, Production, Archived, None"; \
		exit 1; \
	fi
	@echo "Promoting model version $(VERSION) to $(STAGE)..."
	@docker compose run --rm api python scripts/manage_registry.py promote --version $(VERSION) --stage $(STAGE)

registry-info:
	@if [ -z "$(VERSION)$(STAGE)" ]; then \
		echo "Error: VERSION or STAGE required"; \
		echo "Usage: make registry-info VERSION=1"; \
		echo "   or: make registry-info STAGE=Production"; \
		exit 1; \
	fi
	@if [ -n "$(VERSION)" ]; then \
		docker compose run --rm api python scripts/manage_registry.py info --version $(VERSION); \
	else \
		docker compose run --rm api python scripts/manage_registry.py info --stage $(STAGE); \
	fi

mlflow-ui:
	@echo "Starting MLflow UI..."
	@echo "Access at: http://localhost:5000"
	@echo "Press Ctrl+C to stop"
	@echo ""
	@docker compose run --rm -p 5000:5000 --entrypoint mlflow api ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

# =============================================================================
# Service Management
# =============================================================================

up:
	@echo "Starting API service..."
	@docker compose up -d
	@echo "✓ API started at http://localhost:8000"
	@echo ""
	@echo "Useful commands:"
	@echo "  make health   # Check service health"
	@echo "  make test-api # Test prediction"
	@echo "  make logs     # View logs"
	@echo "  make docs     # Open documentation"
	@echo ""

down:
	@echo "Stopping API service..."
	@docker compose down
	@echo "✓ Service stopped"

# =============================================================================
# Monitoring Stack
# =============================================================================

up-monitoring:
	@echo "Starting full monitoring stack..."
	@echo "This includes: API, Prometheus, and Grafana"
	@if [ -z "$$GRAFANA_PASSWORD" ] && ! grep -qE '^GRAFANA_PASSWORD=..*' .env 2>/dev/null; then \
		echo ""; \
		echo "Error: GRAFANA_PASSWORD is not set."; \
		echo "Set a non-default Grafana admin password first, e.g.:"; \
		echo "  echo \"GRAFANA_PASSWORD=\$$(openssl rand -hex 16)\" >> .env"; \
		exit 1; \
	fi
	@docker compose --profile monitoring up -d
	@echo ""
	@echo "✓ All services started!"
	@echo ""
	@echo "Access points:"
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo ""
	@echo "Quick commands:"
	@echo "  make prometheus-ui     # Open Prometheus"
	@echo "  make grafana-ui        # Open Grafana"
	@echo "  make monitoring-status # Check status"
	@echo ""

down-monitoring:
	@echo "Stopping all services (API + monitoring)..."
	@docker compose --profile monitoring down
	@echo "✓ All services stopped"

monitoring-status:
	@echo "Monitoring Stack Status:"
	@echo "=========================================="
	@docker compose ps
	@echo ""
	@echo "Service Health:"
	@echo "------------------------------------------"
	@echo -n "API:        "
	@curl -s -f http://localhost:8000/health > /dev/null 2>&1 && echo "✓ Healthy" || echo "✗ Not responding"
	@echo -n "Prometheus: "
	@curl -s -f http://localhost:9090/-/healthy > /dev/null 2>&1 && echo "✓ Healthy" || echo "✗ Not responding"
	@echo -n "Grafana:    "
	@curl -s -f http://localhost:3000/api/health > /dev/null 2>&1 && echo "✓ Healthy" || echo "✗ Not responding"
	@echo ""

prometheus-ui:
	@echo "Opening Prometheus UI..."
	@command -v xdg-open > /dev/null && xdg-open http://localhost:9090 || \
	 command -v open > /dev/null && open http://localhost:9090 || \
	 echo "Please open http://localhost:9090 in your browser"

grafana-ui:
	@echo "Opening Grafana UI..."
	@echo "Default credentials: admin / admin"
	@command -v xdg-open > /dev/null && xdg-open http://localhost:3000 || \
	 command -v open > /dev/null && open http://localhost:3000 || \
	 echo "Please open http://localhost:3000 in your browser"

restart:
	@echo "Restarting API service..."
	@docker compose restart
	@echo "✓ Service restarted"
	@echo "Note: This picks up model changes without rebuilding the image"

# =============================================================================
# API Operations
# =============================================================================

health:
	@echo "Checking API health (detailed status)..."
	@echo "This endpoint returns 503 if model is not loaded."
	@echo ""
	@curl -s http://localhost:8000/health | python3 -m json.tool || \
		echo "Error: API not responding. Run 'make up' to start the service."

test-liveness:
	@echo "Testing liveness probe (/healthz)..."
	@echo "This should ALWAYS return 200 if the process is alive."
	@echo ""
	@curl -s http://localhost:8000/healthz | python3 -m json.tool || \
		echo "Error: API not responding (process may be dead)."

test-readiness:
	@echo "Testing readiness probe (/readyz)..."
	@echo "This returns 200 if ready to serve traffic, 503 if not."
	@echo ""
	@curl -s http://localhost:8000/readyz | python3 -m json.tool || \
		echo "Error: API not responding."

test-api:
	@echo "Testing prediction endpoint with example request..."
	@echo "Using test_request.json (no authentication)"
	@echo ""
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d @test_request.json | python3 -m json.tool || \
		echo "Error: API not responding. Run 'make up' to start the service."

test-auth:
	@echo "Testing prediction with authentication..."
	@echo "This requires SERVICE_TOKEN to be set in docker-compose.yml"
	@echo ""
	@if [ -z "$$SERVICE_TOKEN" ]; then \
		echo "Error: SERVICE_TOKEN not set."; \
		echo "Set it with: export SERVICE_TOKEN=your-secret-token"; \
		echo "Or enable it in docker-compose.yml and restart the service."; \
		exit 1; \
	fi
	@echo "Testing with token: $$SERVICE_TOKEN"
	@echo ""
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $$SERVICE_TOKEN" \
		-d @test_request.json | python3 -m json.tool || \
		echo "Authentication test failed."

test-validation:
	@echo "Testing error handling and validation..."
	@echo "(/predict takes a SINGLE customer object; /drift takes a batch)"
	@echo ""
	@echo "1. Testing empty request body on /predict (should return 422):"
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{}' -w "\nHTTP Status: %{http_code}\n\n" | head -n 20
	@echo ""
	@echo "2. Testing invalid data type on /predict (should return 422):"
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"tenure": "invalid", "MonthlyCharges": 50}' -w "\nHTTP Status: %{http_code}\n\n" | head -n 20
	@echo ""
	@echo "3. Testing empty batch on /drift (should return 400):"
	@curl -s -X POST http://localhost:8000/drift \
		-H "Content-Type: application/json" \
		-d '{"customers": []}' -w "\nHTTP Status: %{http_code}\n\n" | head -n 20

test-drift:
	@echo "Running comprehensive drift detection tests..."
	@echo "(requires the API to be up: make up)"
	@docker compose run --rm -e API_URL=http://api:8000 api python scripts/test_drift.py

drift-info:
	@echo "Fetching drift detection configuration..."
	@echo ""
	@curl -s http://localhost:8000/drift/info | python3 -m json.tool 2>/dev/null || \
		echo "Error: API not responding. Run 'make up' to start the service."

retrain:
	@echo "Triggering model retraining via API..."
	@echo "Note: This starts retraining in the background."
	@echo "      Monitor logs with 'make logs' to track progress."
	@echo ""
	@curl -s -X POST http://localhost:8000/retrain | python3 -m json.tool 2>/dev/null || \
		echo "Error: API not responding or retraining not allowed (rate limited)."

metrics:
	@echo "Fetching Prometheus metrics..."
	@curl -s http://localhost:8000/metrics/ || \
		echo "Error: API not responding. Run 'make up' to start the service."

docs:
	@echo "Opening API documentation..."
	@command -v xdg-open > /dev/null && xdg-open http://localhost:8000/docs || \
	 command -v open > /dev/null && open http://localhost:8000/docs || \
	 echo "Please open http://localhost:8000/docs in your browser"

# =============================================================================
# Monitoring & Debugging
# =============================================================================

logs:
	@echo "Showing container logs (Ctrl+C to exit)..."
	@docker compose logs -f

shell:
	@echo "Opening shell in container..."
	@docker compose run --rm --entrypoint /bin/bash api

status:
	@echo "=========================================="
	@echo "Churn Prediction Pipeline Status"
	@echo "=========================================="
	@echo ""
	@echo "Docker Services:"
	@docker compose ps
	@echo ""
	@echo "Models:"
	@[ -f models/churn_model_latest.joblib ] && echo "  ✓ models/churn_model_latest.joblib" || echo "  ✗ Model missing (run 'make train')"
	@ls models/metadata_*.json > /dev/null 2>&1 && echo "  ✓ Model metadata" || echo "  ✗ Metadata missing"
	@echo ""
	@echo "Data:"
	@[ -f data/dataset.arff ] && echo "  ✓ data/dataset.arff" || echo "  ✗ Dataset missing"
	@[ -d diagnostics ] && echo "  ✓ diagnostics/" || echo "  ✗ diagnostics/ missing"
	@echo ""
	@echo "Docker Image:"
	@docker images churn-service:latest --format "  ✓ churn-service:latest ({{.Size}})" 2>/dev/null || echo "  ✗ Image not built (run 'make build')"
	@echo ""

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running all tests..."
	@docker compose run --rm --entrypoint pytest api -v
	@echo "✓ All tests complete"

test-unit:
	@echo "Running unit tests..."
	@docker compose run --rm --entrypoint pytest api -v -m unit
	@echo "✓ Unit tests complete"

test-integration:
	@echo "Running integration tests..."
	@docker compose run --rm --entrypoint pytest api -v -m integration
	@echo "✓ Integration tests complete"

test-e2e:
	@echo "Running end-to-end tests..."
	@docker compose run --rm --entrypoint pytest api -v -m e2e
	@echo "✓ End-to-end tests complete"

test-quick:
	@echo "Running fast tests (skipping slow ones)..."
	@docker compose run --rm --entrypoint pytest api -v -m "not slow"
	@echo "✓ Quick tests complete"

test-coverage:
	@echo "Running tests with coverage report..."
	@docker compose run --rm --entrypoint pytest api --cov --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report generated (see htmlcov/index.html)"

test-schema:
	@echo "Testing schema alignment..."
	@docker compose run --rm api python demo_schema_alignment.py

# =============================================================================
# Security Scanning
# =============================================================================

security-scan:
	@echo "Running container security scan..."
	@echo "This scans the Docker image for:"
	@echo "  • OS vulnerabilities"
	@echo "  • Python package vulnerabilities"
	@echo "  • Hardcoded secrets"
	@echo ""
	@bash scripts/security_scan.sh "CRITICAL,HIGH,MEDIUM" "table"
	@echo ""
	@echo "For detailed analysis, see:"
	@echo "  • GitHub Security tab (in CI)"
	@echo "  • Run: make security-scan-critical (for critical/high only)"

security-scan-critical:
	@echo "Scanning for CRITICAL and HIGH severity vulnerabilities only..."
	@echo ""
	@bash scripts/security_scan.sh "CRITICAL,HIGH" "table"

# =============================================================================
# Load Testing
# =============================================================================

load-test:
	@echo "Running load test to validate SLO compliance..."
	@echo ""
	@echo "Configuration:"
	@echo "  • Users: 50 concurrent"
	@echo "  • Spawn rate: 10 users/second"
	@echo "  • Duration: 60 seconds"
	@echo "  • Target: http://api:8000 (the running API service on the compose network)"
	@echo ""
	@echo "Validating SLOs:"
	@echo "  • p95 latency < 500ms"
	@echo "  • p99 latency < 1000ms"
	@echo "  • Error rate < 1%"
	@echo ""
	@docker compose run --rm --entrypoint bash api -c "\
		pip install locust==2.32.3 >/dev/null 2>&1 && \
		cd tests && \
		locust -f locustfile.py \
			--headless \
			--users 50 \
			--spawn-rate 10 \
			--run-time 60s \
			--host http://api:8000 \
			--html /app/tests/locust_report.html \
			--csv /app/tests/locust_results \
	"
	@echo ""
	@echo "✓ Load test complete"
	@echo "📊 Report: tests/locust_report.html"
	@echo ""

load-test-ui:
	@echo "Starting Locust web UI..."
	@echo ""
	@echo "Configuration:"
	@echo "  • Open: http://localhost:8089"
	@echo "  • Target: http://api:8000 (the running API service on the compose network)"
	@echo "  • Suggested settings:"
	@echo "    - Users: 50"
	@echo "    - Spawn rate: 10"
	@echo "    - Host: http://api:8000"
	@echo ""
	@echo "Press Ctrl+C to stop"
	@echo ""
	@docker compose run --rm -p 8089:8089 --entrypoint bash api -c "\
		pip install locust==2.32.3 >/dev/null 2>&1 && \
		cd tests && \
		locust -f locustfile.py --host http://api:8000 \
	"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Stopping services and removing containers..."
	@docker compose down
	@echo "✓ Services stopped"

clean-data:
	@echo "Removing generated diagnostics and logs..."
	@rm -rf diagnostics/*.png diagnostics/*.txt diagnostics/*.csv 2>/dev/null || true
	@rm -rf logs/*.log logs/*.log.* 2>/dev/null || true
	@echo "✓ Generated data removed (models preserved)"

clean-models:
	@echo "Removing trained models..."
	@rm -rf models/*.joblib models/*.json models/*.sha256 models/.last_retrain 2>/dev/null || true
	@echo "✓ Models removed (run 'make train' to retrain)"

clean-all: clean clean-data clean-models
	@echo "Removing Docker images..."
	@docker compose down -v
	@docker rmi churn-service:latest 2>/dev/null || true
	@echo "✓ Complete cleanup finished"
	@echo ""
	@echo "To start fresh, run:"
	@echo "  make setup"
	@echo ""

purge: clean-all
	@echo "=========================================="
	@echo "⚠️  NUCLEAR CLEANUP - REMOVING EVERYTHING"
	@echo "=========================================="
	@echo ""
	@echo "Removing Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✓ Python cache removed"
	@echo ""
	@echo "Removing all log files (including rotated)..."
	@docker compose run --rm --entrypoint /bin/bash api -c "rm -rf /app/logs/*" 2>/dev/null || rm -rf logs/* 2>/dev/null || true
	@echo "✓ All logs removed"
	@echo ""
	@echo "Removing all diagnostics..."
	@docker compose run --rm --entrypoint /bin/bash api -c "rm -rf /app/diagnostics/*" 2>/dev/null || rm -rf diagnostics/* 2>/dev/null || true
	@echo "✓ All diagnostics removed"
	@echo ""
	@echo "Removing all models and metadata..."
	@docker compose run --rm --entrypoint /bin/bash api -c "rm -rf /app/models/*" 2>/dev/null || rm -rf models/* 2>/dev/null || true
	@echo "✓ All models removed"
	@echo ""
	@echo "Checking for any leftover generated files..."
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf htmlcov 2>/dev/null || true
	@echo "✓ Test artifacts removed"
	@echo ""
	@echo "Removing MLflow tracking data..."
	@docker compose run --rm --entrypoint /bin/bash api -c "rm -rf /app/mlruns /app/mlartifacts" 2>/dev/null || rm -rf mlruns mlartifacts 2>/dev/null || true
	@rm -f mlflow.db 2>/dev/null || true
	@echo "✓ MLflow data removed"
	@echo ""
	@echo "Removing runtime configuration files..."
	@rm -rf configs/* 2>/dev/null || true
	@echo "✓ Runtime configs removed"
	@echo ""
	@echo "Removing Python package metadata..."
	@rm -rf *.egg-info 2>/dev/null || true
	@echo "✓ Package metadata removed"
	@echo ""
	@echo "=========================================="
	@echo "✓ PURGE COMPLETE"
	@echo "=========================================="
	@echo ""
	@echo "All generated files have been removed:"
	@echo "  ✓ Docker containers, volumes, images"
	@echo "  ✓ Models and metadata"
	@echo "  ✓ Diagnostics (plots, reports)"
	@echo "  ✓ Logs (including rotated files)"
	@echo "  ✓ Python cache (__pycache__, *.pyc)"
	@echo "  ✓ Test artifacts (.pytest_cache, .coverage)"
	@echo "  ✓ MLflow data (mlruns, mlartifacts, mlflow.db)"
	@echo "  ✓ Python package metadata (*.egg-info)"
	@echo ""
	@echo "The project is now in pristine state!"
	@echo "To start over: make setup"
	@echo ""

# =============================================================================
# Development Shortcuts
# =============================================================================

# Quick iteration: train and restart  
quick-update: train restart
	@echo "✓ Model retrained and service restarted"

# Full rebuild
rebuild: build train up
	@echo "✓ Complete rebuild finished"
