# setup.ps1 - Windows PowerShell Setup Script for RAG Pipeline
# This script provides the same functionality as setup.sh for Windows users

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "RAG Pipeline - Automated Setup (Windows)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor White
Write-Host "  1. Build Docker containers" -ForegroundColor White
Write-Host "  2. Start Ollama LLM service" -ForegroundColor White
Write-Host "  3. Download Llama 3.2 model (~2GB)" -ForegroundColor White
Write-Host "  4. Preprocess 420 documents" -ForegroundColor White
Write-Host "  5. Build vector index (3 chunking strategies)" -ForegroundColor White
Write-Host ""
Write-Host "Estimated time: 5-10 minutes (depending on internet speed)" -ForegroundColor Yellow
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

# Step 1: Build containers
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "📦 Step 1/5: Building Docker containers..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Containers built successfully" -ForegroundColor Green
Write-Host ""

# Step 2: Start Ollama service
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🚀 Step 2/5: Starting Ollama LLM service..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose up -d ollama
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start Ollama service!" -ForegroundColor Red
    exit 1
}
Write-Host "⏳ Waiting for Ollama to be ready (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10
Write-Host "✓ Ollama service running" -ForegroundColor Green
Write-Host ""

# Step 3: Pull LLM model
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "📥 Step 3/5: Downloading LLM model (llama3.2:3b, ~2GB)..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "⏳ This may take a few minutes on first run..." -ForegroundColor Yellow
docker compose exec ollama ollama pull llama3.2:3b
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to download model!" -ForegroundColor Red
    Write-Host "This might be due to:" -ForegroundColor Yellow
    Write-Host "  - Slow internet connection (try again)" -ForegroundColor Yellow
    Write-Host "  - Ollama service not fully started (wait and retry)" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Model downloaded and cached" -ForegroundColor Green
Write-Host ""

# Step 4: Run preprocessing
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "⚙️  Step 4/5: Preprocessing corpus (420 documents)..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose run --rm rag-pipeline preprocess
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Preprocessing failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Preprocessing complete" -ForegroundColor Green
Write-Host ""

# Step 5: Build vector index
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🔍 Step 5/5: Building vector index (3 chunking strategies)..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose run --rm rag-pipeline index
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Indexing failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Vector index built" -ForegroundColor Green
Write-Host ""

# Success message
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "✓ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "The RAG pipeline is ready to use!" -ForegroundColor White
Write-Host ""
Write-Host "Try these commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Query with retrieval only" -ForegroundColor Cyan
Write-Host '  docker compose run --rm rag-pipeline query "How do I use StandardScaler?"' -ForegroundColor White
Write-Host ""
Write-Host "  # Query with answer generation (LLM)" -ForegroundColor Cyan
Write-Host '  docker compose run --rm rag-pipeline query "How do I use StandardScaler?" --generate' -ForegroundColor White
Write-Host ""
Write-Host "  # More examples" -ForegroundColor Cyan
Write-Host '  docker compose run --rm rag-pipeline query "What is PCA?" --generate' -ForegroundColor White
Write-Host '  docker compose run --rm rag-pipeline query "How to handle missing values?" --generate' -ForegroundColor White
Write-Host ""
Write-Host "To stop services:" -ForegroundColor Yellow
Write-Host "  docker compose down" -ForegroundColor White
Write-Host ""
Write-Host "To clean everything:" -ForegroundColor Yellow
Write-Host "  docker compose down -v" -ForegroundColor White
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green

