#!/bin/bash
# setup.sh - Two-stage recommender: one-command quick setup
# Builds the image and runs the fast classical smoke (~5 min). Full three-stage
# reproduction is `make reproduce` (~45 min).

set -e  # Exit on any error

echo "================================================================================"
echo "Two-Stage Recommender - Quick Setup"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Build the Docker image"
echo "  2. Run a full-catalogue evaluation of the four classical models (fast smoke)"
echo ""
echo "Estimated time: ~5 minutes (all three stages: 'make reproduce', ~45 minutes)"
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

# Step 2: Full-catalogue evaluation
echo "================================================================================"
echo "📊 Step 2/2: Evaluating models on the full catalogue (the honest protocol)..."
echo "================================================================================"
docker compose run --rm reclab evaluate
echo "✅ Evaluation complete (tables written to outputs/)"
echo ""

# Success message
echo "================================================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "The recommender pipeline is ready — all three stages are available via make."
echo ""
echo "Try these commands:"
echo ""
echo "  # Sampled-negative evaluation + the protocol-disagreement table"
echo "  make sampled"
echo ""
echo "  # Coverage / Gini / popularity-bias metrics"
echo "  make beyond"
echo ""
echo "  # Reproduce every number in the README end to end, all three stages (~45 min)"
echo "  make reproduce"
echo ""
echo "For help:"
echo "  make help"
echo ""
echo "================================================================================"
