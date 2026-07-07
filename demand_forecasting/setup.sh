#!/bin/bash
# setup.sh - Complete demand forecasting setup
# One-command setup for portfolio demonstration

set -e  # Exit on any error

echo "================================================================================"
echo "Demand Forecasting - Automated Setup"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Build the Docker image"
echo "  2. Run a rolling-origin backtest of the seasonal-naive baseline"
echo ""
echo "Estimated time: ~5 minutes"
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

# Step 2: Baseline backtest
echo "================================================================================"
echo "📈 Step 2/2: Backtesting the seasonal-naive baseline (12 folds x 28 days)..."
echo "================================================================================"
docker compose run --rm demandcast backtest --model seasonal_naive
echo "✅ Baseline backtest complete (predictions + scores written to outputs/)"
echo ""

# Success message
echo "================================================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "The forecasting pipeline is ready to use!"
echo ""
echo "Try these commands:"
echo ""
echo "  # Backtest the global LightGBM (point + P10/P50/P90 quantiles, ~10 min)"
echo "  make backtest ARGS=\"--model lgbm\""
echo ""
echo "  # Backtest SARIMAX on the 8 high-volume SKUs (~10 min)"
echo "  make backtest ARGS=\"--model sarimax\""
echo ""
echo "  # Apples-to-apples: LightGBM on the same 8 SKUs"
echo "  make backtest ARGS=\"--model lgbm --subset sarimax\""
echo ""
echo "  # Estimate the promotion lift (PPML fixed-effects regression, ~5 min)"
echo "  make promo-lift"
echo ""
echo "  # Reproduce every number in the README in one go (~30-40 min)"
echo "  make reproduce"
echo ""
echo "For help:"
echo "  make help"
echo ""
echo "================================================================================"
