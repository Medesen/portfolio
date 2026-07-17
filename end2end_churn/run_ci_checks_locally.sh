#!/bin/bash
# Local CI Checks - Run GitHub Actions tests locally before pushing
# This script mimics EXACTLY what GitHub Actions does
# Usage: ./run_ci_checks_locally.sh [test_type]
#   test_type: lint | test | integration | load | all (default: all)

set -e

TEST_TYPE="${1:-all}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Activate venv if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo ""
fi

echo "==========================================="
echo "Running CI Checks Locally (GitHub Actions Mirror)"
echo "==========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version (GitHub uses 3.12)
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo -e "${YELLOW}⚠️  Warning: GitHub uses Python 3.12, you have $PYTHON_VERSION${NC}"
    echo ""
fi

# Function to run linting (matches GitHub 'lint' job)
run_lint() {
    echo -e "${BLUE}[1/4] Running Linting Checks...${NC}"
    echo "-------------------------------------------"
    
    # Check if tools are available, install to venv if needed
    if ! command -v flake8 &> /dev/null || ! command -v black &> /dev/null || ! command -v isort &> /dev/null; then
        echo -e "${YELLOW}Installing linting tools to venv...${NC}"
        pip install flake8==7.1.1 black==24.10.0 isort==5.13.2
        echo ""
    fi
    
    echo "Running flake8..."
    flake8 . --exclude=venv,mlruns,mlartifacts,.pytest_cache
    
    echo "Running black..."
    black --check . --exclude="venv|mlruns|mlartifacts|.pytest_cache"
    
    echo "Running isort..."
    isort --check-only . --skip venv --skip mlruns --skip mlartifacts --skip .pytest_cache
    
    echo -e "${GREEN}✅ Linting passed!${NC}"
    echo ""
}

# Function to run tests (matches GitHub 'test' job - runs ALL tests with coverage)
run_test() {
    echo -e "${BLUE}[2/4] Running Tests with Coverage...${NC}"
    echo "-------------------------------------------"
    
    # Check if pytest is available, install to venv if needed
    if ! command -v pytest &> /dev/null; then
        echo -e "${YELLOW}Installing pytest and coverage tools to venv...${NC}"
        pip install pytest==8.3.4 pytest-cov==6.0.0 pytest-xdist==3.6.1 httpx==0.28.1
        echo ""
    fi
    
    echo "Running: pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing"
    pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing
    
    echo -e "${GREEN}✅ Tests passed!${NC}"
    echo ""
}

# Function to run integration tests (matches GitHub 'integration-test' job)
run_integration() {
    echo -e "${BLUE}[3/4] Running Integration Tests with Docker...${NC}"
    echo "-------------------------------------------"
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
    
    echo "Starting services with docker compose..."
    docker compose up -d
    
    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 10
    
    # Check health endpoint
    echo "Checking API health..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            echo -e "${GREEN}✅ API is healthy${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ API health check failed${NC}"
            docker compose logs
            docker compose down
            exit 1
        fi
        sleep 2
    done
    
    echo ""
    echo "Running: pytest tests/ -v -m \"not load\""
    pytest tests/ -v -m "not load"
    
    echo -e "${GREEN}✅ Integration tests passed!${NC}"
    echo ""
}

# Function to run load tests (matches GitHub 'load-test' job)
run_load_test() {
    echo -e "${BLUE}[4/4] Running Load Tests...${NC}"
    echo "-------------------------------------------"
    
    # Check if services are running
    if ! curl -f http://localhost:8000/health &> /dev/null; then
        echo "Starting services..."
        docker compose up -d
        sleep 10
    fi
    
    # Check if locust is available, install to venv if needed
    if ! command -v locust &> /dev/null; then
        echo -e "${YELLOW}Installing locust to venv...${NC}"
        pip install locust==2.32.3
        echo ""
    fi
    
    echo "Running Locust load test (50 users, 60s)..."
    cd tests
    locust -f locustfile.py --headless \
        --users 50 --spawn-rate 10 --run-time 60s \
        --host http://localhost:8000 \
        --csv=locust_results \
        --html=locust_report.html
    
    # Validate SLOs (matches GitHub workflow line 395-439)
    echo ""
    echo "Validating SLOs..."
    
    # Check if pandas is available, install to venv if needed
    if ! python3 -c "import pandas" &> /dev/null; then
        echo -e "${YELLOW}Installing pandas to venv...${NC}"
        pip install pandas==2.2.3
        echo ""
    fi
    
    python3 << 'EOF'
import pandas as pd
import sys

try:
    df = pd.read_csv('locust_results_stats.csv')
    
    # Filter for /predict endpoint with POST method
    predict_row = df[(df['Name'] == '/predict') & (df['Type'] == 'POST')]
    
    if predict_row.empty:
        print('❌ No POST /predict requests recorded - cannot validate SLOs')
        sys.exit(1)
    
    # Extract metrics
    p95 = predict_row['95%'].values[0]
    p99 = predict_row['99%'].values[0]
    failure_count = predict_row['Failure Count'].values[0]
    request_count = predict_row['Request Count'].values[0]
    failure_rate = (failure_count / request_count) * 100 if request_count > 0 else 0
    
    # Print results
    print('\n' + '='*70)
    print('SLO VALIDATION RESULTS')
    print('='*70)
    print(f'\n📊 Load Test Metrics:')
    print(f'   Total Requests: {request_count}')
    print(f'   Failed Requests: {failure_count}')
    print(f'   p95 Latency: {p95}ms')
    print(f'   p99 Latency: {p99}ms')
    print(f'   Failure Rate: {failure_rate:.2f}%')
    
    # Validate against SLOs
    p95_pass = p95 < 500
    p99_pass = p99 < 1000
    error_pass = failure_rate < 1.0
    
    print(f'\n📋 SLO Compliance:')
    print(f'   {"✅" if p95_pass else "❌"} p95 < 500ms: {"PASS" if p95_pass else "FAIL"}')
    print(f'   {"✅" if p99_pass else "❌"} p99 < 1000ms: {"PASS" if p99_pass else "FAIL"}')
    print(f'   {"✅" if error_pass else "❌"} Error rate < 1%: {"PASS" if error_pass else "FAIL"}')
    print('='*70)
    
    if not (p95_pass and p99_pass and error_pass):
        print('\n❌ SLO validation failed!')
        sys.exit(1)
    
    print('\n✅ All SLOs met!')
    
except FileNotFoundError:
    print('❌ locust_results_stats.csv not found')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error validating SLOs: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    RESULT=$?
    cd ..
    
    if [ $RESULT -eq 0 ]; then
        echo -e "${GREEN}✅ Load test passed!${NC}"
    else
        echo -e "${RED}❌ Load test failed!${NC}"
        exit 1
    fi
    echo ""
}

# Cleanup function
cleanup() {
    if [ "$KEEP_RUNNING" != "true" ]; then
        echo "Cleaning up..."
        docker compose down
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Main execution
case $TEST_TYPE in
    lint)
        run_lint
        ;;
    test)
        run_test
        ;;
    integration)
        run_integration
        ;;
    load)
        run_load_test
        ;;
    all)
        run_lint
        run_test
        run_integration
        run_load_test
        ;;
    *)
        echo -e "${RED}Invalid test type: $TEST_TYPE${NC}"
        echo "Usage: $0 [lint|test|integration|load|all]"
        exit 1
        ;;
esac

echo -e "${GREEN}==========================================="
echo "✅ All CI checks passed!"
echo "===========================================${NC}"
echo ""
echo "You can now push to GitHub with confidence! 🚀"
