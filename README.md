# ML Portfolio

A collection of machine learning projects I've built to demonstrate practical implementations, systematic evaluation, and production-oriented engineering practices. Each project includes complete documentation, reproducible Docker environments, and real-world datasets.

NB: This is a work in progress. Projects will be added gradually, but probably not very rapidly, as I'm writing this in my spare time.

## Current Projects

### 1. RAG Pipeline - Retrieval System with Systematic Evaluation

**Status:** Complete  
**Domain:** Technical documentation Q&A  
**Key Finding:** Fixed chunking achieved the highest retrieval performance (Recall@10: 0.51, MRR: 0.51)  
**Tech Stack:** ChromaDB, sentence-transformers, Ollama (Llama 3.2), Docker

A retrieval-augmented generation system I built to compare chunking strategies for technical documentation Q&A. I implemented and evaluated three chunking approaches (fixed, semantic, hierarchical) using standard IR metrics across 35 test questions. Everything runs locally in Docker with no API keys required.

**Highlights:**
- Systematic evaluation comparing three chunking strategies
- Complete Docker deployment with automated setup
- 35-question test set with IR metrics (Recall@k, MRR, NDCG)
- Runs entirely locally using Ollama for LLM generation
- ~10 minute setup time, fully reproducible

**[View Project →](rag_pipeline/)**

---

### 2. Customer Churn Prediction - Production ML Pipeline

**Status:** Complete  
**Domain:** Telecom customer churn prediction  
**Best Performance:** ROC AUC 0.837 (held-out test set)  
**Tech Stack:** FastAPI, MLflow, Prometheus/Grafana, Docker, Kubernetes

A production-oriented ML service I built to predict customer churn with complete end-to-end engineering. The project demonstrates training with hyperparameter tuning, REST API serving, drift detection, experiment tracking, monitoring dashboards, comprehensive testing, and CI/CD pipeline. Everything runs via Docker with zero local setup required.

**Highlights:**
- Complete ML service (training → API → monitoring → deployment)
- Three algorithms with automated hyperparameter tuning
- MLflow experiment tracking and model registry
- Drift detection with automatic retraining triggers
- 200+ tests with 91% coverage, full CI/CD pipeline
- Kubernetes deployment manifests with auto-scaling
- ~5 minute setup time, fully reproducible

**[View Project →](end2end_churn/)**

---

### 3. Config-Driven ML - Hydra + Pydantic Configuration Layer

**Status:** Complete  
**Domain:** Configuration management for ML experiments  
**Pattern:** Pydantic schemas as single source of truth → Hydra composition → validated CLI  
**Tech Stack:** Hydra, Pydantic v2, scikit-learn, Docker

A small, config-driven training pipeline I built to demonstrate a pattern I consider essential for experiment-heavy ML work: Hydra for config composition and CLI overrides, Pydantic for schema validation, joined at a single validated boundary. The ML task (regression on scikit-learn's bundled diabetes dataset) is deliberately simple — the configuration layer is the point. Everything runs locally in Docker with no API keys and no downloads at runtime.

**Highlights:**
- Every hyperparameter overridable from the CLI, validated before training starts
- Swappable model families via a config group backed by a discriminated union
- Named experiment configs that inherit from a base and change only what differs
- Multirun sweeps (both models × three seeds) with zero extra code
- Config snapshot saved per run — any run can be re-scored from its output directory (within the same pinned environment)
- ~2 minute setup time, fully reproducible

**[View Project →](config_driven_ml/)**

---

### 4. Demand Forecasting - Promo-Aware Retail Forecasting with Econometric Lift Estimation

**Status:** Complete  
**Domain:** Retail demand forecasting & promotion-effect estimation  
**Key Findings:** Global LightGBM improves MASE by 35% over seasonal-naive across 118 SKUs; SARIMAX wins on the high-volume subset; promotions multiply expected daily sales ≈ 4.6× (PPML, 95% CI [+307%, +429%])  
**Tech Stack:** LightGBM, statsmodels (SARIMAX, PPML), pandas, Docker

A forecasting system I built to compare classical and ML approaches the way time-series work should be evaluated: rolling-origin backtests against a seasonal-naive baseline, never a random split. A single global LightGBM (Tweedie objective, P10/P50/P90 quantiles) covers all 118 daily series; per-series SARIMAX with promotion regressors is fit where its assumptions hold — and wins there, which is the honest headline. A PPML fixed-effects panel regression estimates the promotion lift with cluster-robust inference. The dataset (5 years of daily pasta sales with per-SKU promo flags, CC BY) ships in the repo: no accounts, no API keys.

**Highlights:**
- Rolling-origin evaluation: 12 expanding-window folds × 28-day horizon
- Metrics chosen for zero-inflated count data (MASE/WAPE, not sMAPE) with reasoning on record
- Quantile forecasts with pinball loss and interval-coverage evaluation (0.778 vs 0.80 target)
- PPML promo-lift estimation with SKU fixed effects and stated identification assumptions
- Leakage tests that corrupt the future and assert forecasts don't move
- 27 tests, ~5 minute setup, fully reproducible (`make reproduce` regenerates every README number)

**[View Project →](demand_forecasting/)**

---

### 5. Experimentation & Uplift - A/B Analysis, CUPED, and Targeting on a Randomized Trial

**Status:** Complete  
**Domain:** Marketing experimentation & causal inference (commercial)  
**Key Findings:** On a randomized e-mail trial, the men's e-mail has the larger *average* effect but the women's e-mail carries the real treatment-effect *heterogeneity*; targeting the top 30% by predicted uplift captures ~55% of the incremental visits of mailing everyone; CUPED yields only ~0–3% variance reduction here — an honest null explained by weak pre-period covariates  
**Tech Stack:** scikit-learn, LightGBM (S/T/X-learners), statsmodels, scipy, Docker

An experimentation-and-uplift study I built to work through the three questions a commercial data scientist asks of an experiment: *did it work* (ATE with unpooled CIs, minimum-detectable-effect, and multiplicity control), *can we measure it more cheaply* (CUPED and Lin regression adjustment — reported honestly, including where they don't help), and *whom should we target* (S/T/X-learner uplift models evaluated with Qini curves on a held-out split, turned into a concrete top-k targeting policy). Built on Kevin Hillstrom's classic 64,000-customer randomized e-mail dataset, which ships in the repo: no accounts, no API keys, no downloads.

**Highlights:**
- Randomization verified (max |SMD| 0.014) before any outcome is touched
- ATEs with unpooled CIs, minimum-detectable-effect, and Holm/BH multiplicity control across the arm × outcome grid
- CUPED + Lin (2013) regression adjustment, with the honest ~0–3% variance reduction this data allows — mechanism validated on synthetic data where reduction = ρ² exactly
- S/T/X-learner uplift models evaluated by Qini on a held-out split, with a decile table and a top-k targeting simulation vs response-model and random targeting
- 18 tests incl. estimator-recovery on known synthetic truth, ~5 minute setup, fully reproducible (`make reproduce` regenerates every number and the figure)

**[View Project →](causal_uplift/)**

---

## What This Portfolio Demonstrates

### Engineering Practices

**Reproducibility:** I build all projects using Docker with automated setup scripts. Anyone can clone and run them without manual environment configuration or API dependencies.

**Documentation:** Each project includes comprehensive documentation covering architecture, design decisions, and iteration history. I write README files with quick-start guides, troubleshooting sections, and FAQs to make projects immediately usable.

**Evaluation:** I emphasize systematic evaluation using standard metrics. For example, the RAG pipeline uses industry-standard IR metrics (Recall@k, MRR, NDCG) rather than anecdotal assessment.

### Technical Skills

**ML/AI:** Retrieval-augmented generation, classification models (Random Forest, XGBoost, Logistic Regression), embedding models, vector databases, LLM integration, hyperparameter tuning, threshold optimization, drift detection, evaluation frameworks

**Software Engineering:** REST APIs (FastAPI), modular architecture, dependency injection, configuration management (Hydra + Pydantic), state tracking, logging, type hints, comprehensive testing (unit/integration/e2e)

**MLOps:** MLflow experiment tracking, model registry, CI/CD pipelines (GitHub Actions), security scanning, load testing, Kubernetes deployment

**Monitoring:** Prometheus metrics, Grafana dashboards, request tracing, SLO tracking

**DevOps:** Docker & Docker Compose, automated deployment, multi-service orchestration, cross-platform compatibility (Linux/macOS/Windows)

**Data Engineering:** HTML parsing, chunking strategies, data preprocessing pipelines, batch processing, feature engineering

---

## Repository Structure

```
portfolio/
├── README.md                    # Portfolio overview (this file)
├── rag_pipeline/                # Project 1: RAG system with evaluation
│   ├── README.md               # Complete documentation
│   ├── ARCHITECTURE.md         # System design patterns
│   ├── DESIGN.md               # Design decisions & trade-offs
│   ├── CHANGELOG.md            # Development iterations (1-5)
│   ├── src/                    # Source code (31 Python modules)
│   │   ├── preprocessing/      # HTML → JSON pipeline
│   │   ├── chunking/          # 3 chunking strategies
│   │   ├── retrieval/         # Embeddings & vector search
│   │   ├── generation/        # LLM integration
│   │   ├── evaluation/        # Metrics & analysis
│   │   └── utils/             # Config, logging
│   ├── tests/                  # 103 unit tests
│   ├── data/
│   │   ├── corpus/            # Scikit-learn docs (420 files, tracked)
│   │   └── evaluation/        # Test set (35 Q&A pairs, tracked)
│   ├── config/
│   │   └── config.yaml        # YAML configuration
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Multi-service orchestration
│   ├── Makefile               # 20+ command shortcuts
│   └── setup.sh / setup.ps1   # Automated setup scripts
├── end2end_churn/              # Project 2: Production ML pipeline
│   ├── README.md               # Complete documentation
│   ├── src/                    # Source code
│   │   ├── api/               # FastAPI service
│   │   ├── data/              # Data loading & preprocessing
│   │   ├── models/            # Model pipelines & factory
│   │   ├── training/          # Training & tuning
│   │   ├── evaluation/        # Metrics & visualizations
│   │   └── utils/             # Logging, I/O, metrics, drift
│   ├── tests/                  # 200+ tests (91% coverage)
│   ├── k8s/                   # Kubernetes manifests
│   ├── grafana/               # Grafana dashboards
│   ├── config/                # Training configurations
│   ├── data/                  # Training data
│   ├── Dockerfile             # Container definition
│   ├── docker-compose.yml     # Orchestration (API, MLflow, monitoring)
│   └── Makefile              # 40+ command shortcuts
├── config_driven_ml/           # Project 3: Hydra + Pydantic config layer
│   ├── README.md               # Complete documentation
│   ├── src/mlctl/              # Source code
│   │   ├── config_layer.py    # The bridge: ConfigStore registration + validated boundary
│   │   ├── config_models.py   # Pydantic schemas (single source of truth)
│   │   ├── configs/           # Base + named experiment YAML configs
│   │   ├── pipeline.py        # Train/evaluate logic
│   │   └── main.py            # CLI router
│   ├── tests/                  # 15 tests (schema, composition, roundtrip)
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Single-service orchestration
│   ├── Makefile               # Command shortcuts
│   └── setup.sh / setup.ps1   # Automated setup scripts
├── demand_forecasting/         # Project 4: Forecasting + promo-lift estimation
│   ├── README.md               # Complete documentation
│   ├── DATA_NOTES.md           # EDA findings → design decisions
│   ├── src/demandcast/         # Source code
│   │   ├── data/              # Loader w/ validation, Italian holiday calendar
│   │   ├── evaluation/        # Rolling-origin folds, backtest runner, metrics
│   │   ├── models/            # Baselines, SARIMAX, global LightGBM + quantiles
│   │   ├── analysis/          # PPML promo-lift, forecast plots
│   │   └── main.py            # CLI: backtest / promo-lift / plot-forecast
│   ├── tests/                  # 27 tests (leakage, estimator recovery, metrics)
│   ├── data/raw/               # Bundled dataset (872 KB, CC BY) + provenance
│   ├── assets/                 # Example forecast figure
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Single-service orchestration
│   ├── Makefile               # Command shortcuts incl. `make reproduce`
│   └── setup.sh / setup.ps1   # Automated setup scripts
├── causal_uplift/              # Project 5: Experimentation, CUPED & uplift modeling
│   ├── README.md               # Complete documentation
│   ├── DATA_NOTES.md           # EDA findings → design decisions
│   ├── src/upliftlab/          # Source code
│   │   ├── data/              # Loader + validation; synthetic RCT generator
│   │   ├── experiment/        # Balance (SMD), ATE inference, CUPED / adjustment
│   │   ├── uplift/            # S/T/X-learners; Qini, deciles, targeting
│   │   └── main.py            # CLI: balance / ate / cuped / uplift / all
│   ├── tests/                  # 18 tests incl. estimator-recovery on known truth
│   ├── data/raw/               # Bundled dataset (3.96 MB) + provenance
│   ├── assets/                 # Qini curve figure
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Single-service orchestration
│   ├── Makefile               # Command shortcuts incl. `make reproduce`
│   └── setup.sh / setup.ps1   # Automated setup scripts
└── [future projects...]        # More projects coming soon
```

Each project is self-contained with its own:
- Detailed README with quick-start guide and examples
- Architecture and design documentation (where applicable)
- YAML-based configuration
- Comprehensive test suite with pytest
- Docker environment with automated setup
- Complete dataset (where feasible and permitted by licensing)

---

## Future Projects

This portfolio is actively expanding. Planned additions include:

**LLM / AI / NLP:**
- **PEFT fine-tuning** - QLoRA domain adaptation, LoRA adapters, zero-shot comparison
- **LLM inference service** - FastAPI + vLLM, throughput/latency benchmarking, streaming
- **Prompt engineering cookbook** - Maintainable prompts, guardrails, regression tests
- **Classic NLP baselines** - TF-IDF, BiLSTM-CRF vs Transformers comparison

**Classic ML:**
- **Tabular ML** - Feature engineering, experiment tracking, SHAP, model cards
- **Recommender system** - Two-tower retrieval + ranking, offline metrics
- **Data pipeline orchestration** - Airflow/Prefect, data quality, versioning
- **Analytics & storytelling** - SQL, interactive viz, insights reports

Each new project will follow the same principles:
- Real-world datasets and problems
- Systematic evaluation and comparison
- Complete reproducibility
- Production-oriented engineering
- Comprehensive documentation

---

## Prerequisites

All projects in this portfolio use Docker for reproducibility and consistent environments across platforms:

- **Docker Desktop** ([Get Docker](https://docs.docker.com/get-docker/))
  - Includes Docker Compose (no separate install needed)
  - Works on Linux, macOS, and Windows
  - **Note:** Projects require Docker Compose V2 (`docker compose` command). If you have an older installation with only V1 (`docker-compose` with hyphen), either upgrade Docker or see project-specific troubleshooting sections for workarounds.
- **RAM:** 8GB minimum (12GB recommended)
- **Disk Space:** ~10GB free per project
- **Git** for cloning the repository

Each project's README contains specific prerequisites and platform-specific setup notes.

---

## Getting Started

### Quick Start

Each project uses `make` commands for all operations, providing a consistent interface across projects. Here's the general workflow:

```bash
# Clone the repository
git clone https://github.com/Medesen/portfolio.git
cd portfolio

# Navigate to a specific project
cd rag_pipeline

# Run automated setup
make setup        # Linux/macOS/WSL2/Git Bash (requires Make)
.\setup.ps1       # Windows PowerShell (no Make required)

# The setup process will:
# - Build Docker containers
# - Download required models
# - Process datasets
# - Initialize the environment
```

### What Happens During Setup

Each project's setup script automates the entire environment configuration:
- **Container builds** (~2-3 minutes) - Creates isolated Docker environment
- **Model downloads** (varies by project) - Downloads required ML models
- **Data processing** (~1-2 minutes) - Prepares datasets for use
- **Verification** - Ensures everything is ready to run

After setup completes, you can immediately start using the project. All projects use Docker to ensure consistent environments across Linux, macOS, and Windows.

### Platform-Specific Notes

**Linux & macOS:** All commands work as shown. Docker and Make are typically pre-installed or easily available.

**Windows:** Projects include PowerShell setup scripts (`.ps1`) that work out of the box. Some projects use `make` commands for convenience, which requires installation (`choco install make`) or you can use direct Docker commands. See individual project READMEs for platform-specific details.

---

## Project Philosophy

### Why These Projects?

I focus on projects that demonstrate:

1. **Technical depth** - Not toy examples, but systems that address real challenges
2. **Engineering rigor** - Production-oriented code with testing, logging, and error handling
3. **Systematic evaluation** - Quantitative comparison using standard metrics
4. **Reproducibility** - Anyone can run them without API keys or complex setup
5. **Clear documentation** - Both technical details and high-level rationale

### Design Principles

**Local-first:** I design projects to run locally when feasible, eliminating API costs and ensuring reproducibility. Where cloud APIs would be used in production, I document the trade-offs explicitly.

**Data included:** I include datasets in the repository where licensing and size permit. This maximizes reproducibility and reduces setup friction.

**Honest trade-offs:** I explicitly cover decisions, limitations, and what would change for production deployment in each project's documentation. See each project's DESIGN.md for details.

**Iterative development:** I document the development process in CHANGELOG files, including pivots and lessons learned. This shows real-world problem-solving, not just polished final results.

---

## Technologies Used So Far

Based on the completed projects, I've demonstrated proficiency with:

**ML/AI Frameworks:**
- scikit-learn (Random Forest, Logistic Regression)
- XGBoost (gradient boosting)
- LightGBM (Tweedie & quantile objectives, native API)
- statsmodels (SARIMAX state-space models, PPML/GLM panel regression)
- Sentence-transformers (embedding models)
- ChromaDB (vector database)
- Ollama (local LLM inference)
- PyTorch (underlying framework)

**Python Ecosystem:**
- FastAPI (REST API development)
- Pydantic (data validation, config schemas with discriminated unions)
- Hydra (config composition, CLI overrides, multirun sweeps)
- Type hints and modern Python features
- pytest for comprehensive testing (unit/integration/e2e)
- YAML-based configuration management
- Structured logging

**MLOps & Monitoring:**
- MLflow (experiment tracking, model registry)
- Prometheus (metrics collection)
- Grafana (dashboards and visualization)
- slowapi (rate limiting)

**Development Tools:**
- Docker & Docker Compose for containerization
- Kubernetes (deployment manifests, auto-scaling)
- GitHub Actions (CI/CD pipelines)
- Trivy (security scanning)
- Locust (load testing)
- Git version control
- Make for task automation
- Cross-platform compatibility (Linux/macOS/Windows)

**Evaluation & Metrics:**
- Classification metrics (ROC AUC, precision, recall, F1)
- Information Retrieval metrics (Recall@k, MRR, NDCG)
- Forecasting metrics (MASE, WAPE, pinball loss, interval coverage) with rolling-origin backtesting
- Econometric inference (fixed-effects panel regression, cluster-robust SEs)
- Causal inference & experimentation (randomized ATE, CUPED variance reduction, uplift/Qini, Holm/BH multiplicity control)
- Drift detection (PSI, distribution monitoring)
- Systematic comparative evaluation
- Test set design and curation

As I add more projects, this section will expand to include additional technologies and frameworks.

---

## About This Portfolio

This portfolio showcases my approach to machine learning engineering: combining algorithmic understanding with software engineering best practices. I emphasize reproducibility, systematic evaluation, and clear documentation in every project.

I built these projects to demonstrate end-to-end capability - from problem definition and data processing through implementation, evaluation, and deployment. My focus is on production-oriented engineering rather than research prototypes.

---

## Contact & Links

- **GitHub:** [github.com/Medesen](https://github.com/Medesen)
- **Portfolio Repository:** [portfolio](https://github.com/Medesen/portfolio)

---

**Last Updated:** July 2026  
**Current Projects:** 5 complete, more coming soon  
**License:** MIT — see [LICENSE](LICENSE). Bundled datasets carry their own terms, noted in each project's data documentation.

