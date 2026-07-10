# setup.ps1 - Windows PowerShell Setup Script for Experimentation & Uplift
# This script provides the same functionality as setup.sh for Windows users

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Experimentation & Uplift - Automated Setup (Windows)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor White
Write-Host "  1. Build the Docker image" -ForegroundColor White
Write-Host "  2. Run the covariate-balance check + average treatment effects" -ForegroundColor White
Write-Host ""
Write-Host "Estimated time: ~3 minutes" -ForegroundColor Yellow
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

# Step 2: Balance + ATE
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🧪 Step 2/2: Checking randomization and estimating treatment effects..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose run --rm upliftlab balance
docker compose run --rm upliftlab ate
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: A/B analysis failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ A/B analysis complete (outputs/ has the balance + ATE tables)" -ForegroundColor Green
Write-Host ""

# Success message
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "The experimentation & uplift pipeline is ready to use!" -ForegroundColor White
Write-Host ""
Write-Host "Try these commands:" -ForegroundColor White
Write-Host ""
Write-Host "  # CUPED / regression-adjustment variance reduction" -ForegroundColor Gray
Write-Host "  docker compose run --rm upliftlab cuped" -ForegroundColor White
Write-Host ""
Write-Host "  # Uplift models + Qini evaluation + targeting simulation (~2 min)" -ForegroundColor Gray
Write-Host "  docker compose run --rm upliftlab uplift" -ForegroundColor White
Write-Host ""
Write-Host "  # Reproduce every number and figure in the README in one go (~5 min)" -ForegroundColor Gray
Write-Host "  docker compose run --rm upliftlab all" -ForegroundColor White
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
