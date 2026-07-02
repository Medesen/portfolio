# setup.ps1 - Windows PowerShell Setup Script for Config-Driven ML
# This script provides the same functionality as setup.sh for Windows users

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Config-Driven ML - Automated Setup (Windows)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor White
Write-Host "  1. Build the Docker image" -ForegroundColor White
Write-Host "  2. Train a baseline model (GBM on the diabetes dataset)" -ForegroundColor White
Write-Host ""
Write-Host "Estimated time: ~2 minutes" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting in 3 seconds... (Ctrl+C to cancel)" -ForegroundColor Yellow
Start-Sleep -Seconds 3
Write-Host ""

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker ps | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
if (-not (Test-DockerRunning)) {
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}
Write-Host "Docker is running ✓" -ForegroundColor Green
Write-Host ""

# Create the outputs mount point; if Docker creates it, permissions can be wrong
if (-not (Test-Path "outputs")) {
    New-Item -ItemType Directory -Path "outputs" | Out-Null
}

# Step 1: Build image
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "📦 Step 1/2: Building Docker image..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Image built successfully" -ForegroundColor Green
Write-Host ""

# Step 2: Train baseline model
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🤖 Step 2/2: Training baseline model (GBM, seed 42)..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose run --rm mlctl train
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Training failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Model trained and saved" -ForegroundColor Green
Write-Host ""

# Success message
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "The config-driven pipeline is ready to use!" -ForegroundColor White
Write-Host ""
Write-Host "Try these commands:" -ForegroundColor White
Write-Host ""
Write-Host "  # Swap the model family and override hyperparameters from the CLI" -ForegroundColor Gray
Write-Host "  docker compose run --rm mlctl train model=ridge model.alpha=5.0" -ForegroundColor White
Write-Host ""
Write-Host "  # Run a stored experiment config" -ForegroundColor Gray
Write-Host "  docker compose run --rm mlctl train --config-name=gbm_tuned" -ForegroundColor White
Write-Host ""
Write-Host "  # Sweep both models across three seeds" -ForegroundColor Gray
Write-Host "  docker compose run --rm mlctl train -m model=ridge,gbm seed=0,1,2 experiment_name=sweep" -ForegroundColor White
Write-Host ""
Write-Host "  # Re-score a finished run from its config snapshot" -ForegroundColor Gray
Write-Host "  docker compose run --rm mlctl evaluate run_dir=outputs/baseline/gbm/seed_42" -ForegroundColor White
Write-Host ""
Write-Host "  # Watch validation reject a bad value before training starts" -ForegroundColor Gray
Write-Host "  docker compose run --rm mlctl train model.max_iter=-5" -ForegroundColor White
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
