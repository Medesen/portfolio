#!/bin/bash
# Container Security Scanning Script
# 
# This script runs Trivy vulnerability scanner on the churn-service Docker image
# to identify security vulnerabilities in the base image, OS packages, and Python dependencies.
#
# Usage (positional arguments):
#   ./scripts/security_scan.sh [SEVERITY] [FORMAT]
#   e.g. ./scripts/security_scan.sh CRITICAL,HIGH table
#   Defaults: SEVERITY=CRITICAL,HIGH  FORMAT=table
#
# Container Security Scanning

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
IMAGE_NAME="churn-service:latest"
SEVERITY="${1:-CRITICAL,HIGH}"
FORMAT="${2:-table}"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Container Security Scan${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Image not found: $IMAGE_NAME${NC}"
    echo "Building image..."
    docker compose build api
    echo -e "${GREEN}✓ Image built${NC}"
    echo ""
fi

# Check if Trivy is installed
if ! command -v trivy &> /dev/null; then
    echo -e "${YELLOW}⚠️  Trivy not installed${NC}"
    echo "Running Trivy via Docker..."
    
    # Run Trivy via Docker
    echo -e "${BLUE}Scanning image: $IMAGE_NAME${NC}"
    echo -e "${BLUE}Severity: $SEVERITY${NC}"
    echo -e "${BLUE}Format: $FORMAT${NC}"
    echo ""
    
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        aquasec/trivy:latest image \
        --severity "$SEVERITY" \
        --format "$FORMAT" \
        "$IMAGE_NAME"
else
    # Run Trivy locally
    echo -e "${BLUE}Scanning image: $IMAGE_NAME${NC}"
    echo -e "${BLUE}Severity: $SEVERITY${NC}"
    echo -e "${BLUE}Format: $FORMAT${NC}"
    echo ""
    
    trivy image \
        --severity "$SEVERITY" \
        --format "$FORMAT" \
        "$IMAGE_NAME"
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Additional Scans${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Scan for secrets in the image
echo -e "${BLUE}Scanning for hardcoded secrets...${NC}"
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy:latest image \
    --scanners secret \
    "$IMAGE_NAME" || true

echo ""

# Scan requirements.txt for known vulnerabilities
echo -e "${BLUE}Scanning Python dependencies (requirements.txt)...${NC}"
docker run --rm -v "$PWD:/scan" \
    aquasec/trivy:latest fs \
    --scanners vuln \
    --severity "$SEVERITY" \
    /scan/requirements.txt || true

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Security Scan Complete${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "📊 View detailed results in GitHub Security tab (CI only)"
echo "📋 Run 'docker run --rm aquasec/trivy:latest image --help' for more options"
echo ""
echo "Recommended actions:"
echo "  1. Review CRITICAL and HIGH severity vulnerabilities"
echo "  2. Update base image if vulnerabilities found (python:3.12-slim)"
echo "  3. Update Python dependencies in requirements.txt"
echo "  4. Re-scan after applying fixes"
echo ""

