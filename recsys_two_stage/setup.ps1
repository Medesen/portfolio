# setup.ps1 - Windows PowerShell Setup Script for Two-Stage Recommender (Stage 1)
# This script provides the same functionality as setup.sh for Windows users

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Two-Stage Recommender (Stage 1) - Automated Setup (Windows)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor White
Write-Host "  1. Build the Docker image" -ForegroundColor White
Write-Host "  2. Run a full-catalogue evaluation of the four classical models" -ForegroundColor White
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

# Step 2: Full-catalogue evaluation
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "📊 Step 2/2: Evaluating models on the full catalogue (the honest protocol)..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose run --rm reclab evaluate
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Evaluation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Evaluation complete (outputs/ has the metrics tables)" -ForegroundColor Green
Write-Host ""

# Success message
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "The Stage 1 recommender evaluation pipeline is ready to use!" -ForegroundColor White
Write-Host ""
Write-Host "Try these commands:" -ForegroundColor White
Write-Host ""
Write-Host "  # Sampled-negative evaluation + the protocol-disagreement table" -ForegroundColor Gray
Write-Host "  docker compose run --rm reclab sampled" -ForegroundColor White
Write-Host ""
Write-Host "  # Coverage / Gini / popularity-bias metrics" -ForegroundColor Gray
Write-Host "  docker compose run --rm reclab beyond" -ForegroundColor White
Write-Host ""
Write-Host "  # Reproduce every number in the README in one go (~15 min)" -ForegroundColor Gray
Write-Host "  docker compose run --rm reclab all" -ForegroundColor White
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
