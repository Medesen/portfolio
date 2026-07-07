# setup.ps1 - Windows PowerShell Setup Script for Demand Forecasting
# This script provides the same functionality as setup.sh for Windows users

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Demand Forecasting - Automated Setup (Windows)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor White
Write-Host "  1. Build the Docker image" -ForegroundColor White
Write-Host "  2. Run a rolling-origin backtest of the seasonal-naive baseline" -ForegroundColor White
Write-Host ""
Write-Host "Estimated time: ~5 minutes" -ForegroundColor Yellow
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

# Step 2: Baseline backtest
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "📈 Step 2/2: Backtesting the seasonal-naive baseline..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose run --rm demandcast backtest --model seasonal_naive
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Backtest failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Baseline backtest complete (outputs/ has predictions + scores)" -ForegroundColor Green
Write-Host ""

# Success message
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "The forecasting pipeline is ready to use!" -ForegroundColor White
Write-Host ""
Write-Host "Try these commands:" -ForegroundColor White
Write-Host ""
Write-Host "  # Backtest the global LightGBM (point + P10/P50/P90 quantiles, ~10 min)" -ForegroundColor Gray
Write-Host "  docker compose run --rm demandcast backtest --model lgbm" -ForegroundColor White
Write-Host ""
Write-Host "  # Backtest SARIMAX on the 8 high-volume SKUs (~10 min)" -ForegroundColor Gray
Write-Host "  docker compose run --rm demandcast backtest --model sarimax" -ForegroundColor White
Write-Host ""
Write-Host "  # Apples-to-apples: LightGBM on the same 8 SKUs" -ForegroundColor Gray
Write-Host "  docker compose run --rm demandcast backtest --model lgbm --subset sarimax" -ForegroundColor White
Write-Host ""
Write-Host "  # Estimate the promotion lift (PPML fixed-effects regression, ~5 min)" -ForegroundColor Gray
Write-Host "  docker compose run --rm demandcast promo-lift" -ForegroundColor White
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
