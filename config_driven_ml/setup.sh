#!/bin/bash
# setup.sh - Complete config-driven ML setup
# One-command setup for portfolio demonstration

set -e  # Exit on any error

echo "================================================================================"
echo "Config-Driven ML - Automated Setup"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Build the Docker image"
echo "  2. Train a baseline model (GBM on the diabetes dataset)"
echo ""
echo "Estimated time: ~2 minutes"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3
echo ""

# Check if Docker is running
echo "Checking Docker..."
if ! docker ps > /dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
echo "✅ Docker is running"
echo ""

# Match the container user to the host user so files written to the mounted
# outputs/ directory are owned by you, not root (or a mismatched UID).
export UID 2>/dev/null || true
export GID="${GID:-$(id -g)}"

# Create the outputs mount point host-side; if Docker creates it, it's root-owned
mkdir -p outputs

# Step 1: Build image
echo "================================================================================"
echo "📦 Step 1/2: Building Docker image..."
echo "================================================================================"
docker compose build
echo "✅ Image built successfully"
echo ""

# Step 2: Train baseline model
echo "================================================================================"
echo "🤖 Step 2/2: Training baseline model (GBM, seed 42)..."
echo "================================================================================"
docker compose run --rm mlctl train
echo "✅ Model trained and saved"
echo ""

# Success message
echo "================================================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "The config-driven pipeline is ready to use!"
echo ""
echo "Try these commands:"
echo ""
echo "  # Swap the model family and override hyperparameters from the CLI"
echo "  make train ARGS=\"model=ridge model.alpha=5.0\""
echo ""
echo "  # Run a stored experiment config"
echo "  make train ARGS=\"--config-name=gbm_tuned\""
echo ""
echo "  # Sweep both models across three seeds"
echo "  make sweep"
echo ""
echo "  # Re-score a finished run from its config snapshot"
echo "  make evaluate RUN=outputs/baseline/gbm/seed_42"
echo ""
echo "  # Watch validation reject a bad value before training starts"
echo "  make train ARGS=\"model.max_iter=-5\""
echo ""
echo "For help:"
echo "  make help"
echo ""
echo "================================================================================"
