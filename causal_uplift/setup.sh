#!/bin/bash
# setup.sh - Complete experimentation & uplift setup
# One-command setup for portfolio demonstration

set -e  # Exit on any error

echo "================================================================================"
echo "Experimentation & Uplift - Automated Setup"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Build the Docker image"
echo "  2. Run the covariate-balance check + average treatment effects"
echo ""
echo "Estimated time: ~3 minutes"
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

# Step 2: Balance + ATE
echo "================================================================================"
echo "🧪 Step 2/2: Checking randomization and estimating treatment effects..."
echo "================================================================================"
docker compose run --rm upliftlab balance
docker compose run --rm upliftlab ate
echo "✅ A/B analysis complete (tables written to outputs/)"
echo ""

# Success message
echo "================================================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "The experimentation & uplift pipeline is ready to use!"
echo ""
echo "Try these commands:"
echo ""
echo "  # CUPED / regression-adjustment variance reduction"
echo "  make cuped"
echo ""
echo "  # Uplift models + Qini evaluation + targeting simulation (~2 min)"
echo "  make uplift"
echo ""
echo "  # Reproduce every number and figure in the README in one go (~5 min)"
echo "  make reproduce"
echo ""
echo "For help:"
echo "  make help"
echo ""
echo "================================================================================"
