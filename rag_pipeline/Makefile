# Makefile for RAG Pipeline
# Provides convenient shortcuts for Docker commands

# PHONY targets are not actual files - they're just command names
# This tells Make to always run these commands, even if a file with that name exists
.PHONY: help setup clean build test test-cov test-file \
        preprocess index query eval benchmark \
        up down logs shell \
        clean-docker clean-data clean-all

# Default target - show help when running 'make' with no arguments
help:
	@echo "=========================================="
	@echo "RAG Pipeline - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup & Deployment:"
	@echo "  make setup          - Run complete setup (builds, downloads, indexes)"
	@echo "  make build          - Build Docker containers"
	@echo "  make up             - Start services (Ollama)"
	@echo "  make down           - Stop services"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all unit tests"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make test-file F=<file> - Run specific test file (e.g., F=test_config.py)"
	@echo ""
	@echo "RAG Pipeline Operations:"
	@echo "  make preprocess     - Preprocess corpus (420 documents)"
	@echo "  make index          - Build vector index (all strategies)"
	@echo "  make query Q=\"...\" - Query the system and generate answer"
	@echo "  make query-retrieve Q=\"...\" - Retrieval only (no LLM, for debugging)"
	@echo "  make eval           - Run evaluation framework"
	@echo "  make benchmark      - Benchmark reranking overfetch values"
	@echo ""
	@echo "Utilities:"
	@echo "  make logs           - Show Ollama service logs"
	@echo "  make shell          - Open shell in rag-pipeline container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Stop services and remove containers"
	@echo "  make clean-data     - Remove generated data (processed, vector store, logs)"
	@echo "  make clean-docker   - Remove Docker volumes (models will need re-download)"
	@echo "  make clean-all      - Complete cleanup (everything)"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make query Q=\"How do I use StandardScaler?\""
	@echo "  make query Q=\"What is PCA?\" ARGS=\"--strategy fixed --top-k 5\""
	@echo "  make test-file F=test_metrics.py"
	@echo ""

# =============================================================================
# Setup & Deployment
# =============================================================================

setup:
	@echo "Running complete setup..."
	@./setup.sh

build:
	@echo "Building Docker containers..."
	@docker compose build

up:
	@echo "Starting services..."
	@docker compose up -d

down:
	@echo "Stopping services..."
	@docker compose down

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running all tests..."
	@docker compose run --rm --entrypoint pytest rag-pipeline tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	@docker compose run --rm --entrypoint pytest rag-pipeline tests/ -v --cov=src --cov-report=term-missing

# Run specific test file: make test-file F=test_config.py
test-file:
	@echo "Running tests in $(F)..."
	@docker compose run --rm --entrypoint pytest rag-pipeline tests/$(F) -v

# =============================================================================
# RAG Pipeline Operations
# =============================================================================

preprocess:
	@echo "Preprocessing corpus..."
	@docker compose run --rm rag-pipeline preprocess

preprocess-force:
	@echo "Force reprocessing corpus..."
	@docker compose run --rm rag-pipeline preprocess --force

index:
	@echo "Building vector index..."
	@docker compose run --rm rag-pipeline index

index-force:
	@echo "Force rebuilding vector index..."
	@docker compose run --rm rag-pipeline index --force

# Query with answer generation (default behavior)
# Usage: make query Q="How do I use StandardScaler?"
# With args: make query Q="What is PCA?" ARGS="--strategy fixed --top-k 5"
query:
ifndef Q
	@echo "Error: Please provide a query using Q=\"your question\""
	@echo "Example: make query Q=\"How do I use StandardScaler?\""
	@echo "With options: make query Q=\"What is PCA?\" ARGS=\"--strategy fixed --top-k 10\""
	@exit 1
endif
	@docker compose run --rm rag-pipeline query "$(Q)" --generate $(ARGS)

# Retrieval-only query (for debugging, no LLM generation)
query-retrieve:
ifndef Q
	@echo "Error: Please provide a query using Q=\"your question\""
	@echo "Example: make query-retrieve Q=\"How do I use StandardScaler?\""
	@exit 1
endif
	@docker compose run --rm rag-pipeline query "$(Q)" $(ARGS)

# Run evaluation
eval:
	@echo "Running evaluation framework..."
	@docker compose run --rm rag-pipeline evaluate --report

eval-quick:
	@echo "Running quick evaluation (5 questions)..."
	@docker compose run --rm rag-pipeline evaluate --max-questions 5 --report

# Benchmark reranking overfetch values
# Usage: make benchmark
# With args: make benchmark ARGS="--overfetch-values 30 50 60"
# Or: make benchmark ARGS="--strategy fixed --max-questions 10"
benchmark:
	@echo "Running reranking overfetch benchmark..."
	@docker compose run --rm rag-pipeline python scripts/benchmark_overfetch.py $(ARGS)

# =============================================================================
# Utilities
# =============================================================================

logs:
	@echo "Showing Ollama logs (Ctrl+C to exit)..."
	@docker compose logs -f ollama

shell:
	@echo "Opening shell in rag-pipeline container..."
	@docker compose run --rm --entrypoint /bin/bash rag-pipeline

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Stopping services and removing containers..."
	@docker compose down
	@echo "✓ Services stopped"

clean-data:
	@echo "Removing generated data..."
	@rm -rf data/processed/
	@rm -rf data/state/
	@rm -rf data/vector_store/
	@rm -rf logs/
	@echo "✓ Generated data removed (corpus preserved)"

clean-docker:
	@echo "Removing Docker volumes (models will need re-download)..."
	@docker compose down -v
	@echo "✓ Docker volumes removed"

clean-all: clean clean-data clean-docker
	@echo "✓ Complete cleanup finished"
	@echo "Run 'make setup' to start fresh"

# =============================================================================
# Development Utilities
# =============================================================================

# Check Docker is running
check-docker:
	@docker ps > /dev/null 2>&1 || (echo "Error: Docker is not running" && exit 1)

# Show project status
status:
	@echo "=========================================="
	@echo "RAG Pipeline Status"
	@echo "=========================================="
	@echo ""
	@echo "Docker Services:"
	@docker compose ps
	@echo ""
	@echo "Data Directories:"
	@[ -d data/processed ] && echo "  ✓ data/processed/ exists" || echo "  ✗ data/processed/ missing (run 'make preprocess')"
	@[ -d data/vector_store ] && echo "  ✓ data/vector_store/ exists" || echo "  ✗ data/vector_store/ missing (run 'make index')"
	@[ -d data/corpus/scikit-learn-1.7.2-docs ] && echo "  ✓ Corpus exists" || echo "  ✗ Corpus missing"
	@echo ""

