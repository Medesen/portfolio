# Customer Churn Prediction API - Dockerfile
# Uses Python 3.12 slim image for smaller size

FROM python:3.12-slim

# Accept UID/GID as build arguments (defaults to 1000:1000)
ARG UID=1000
ARG GID=1000
ARG NB_USER=appuser

# Make IDs available to the entrypoint at runtime
ENV NB_USER=${NB_USER} NB_UID=${UID} NB_GID=${GID}

# Set working directory in container
WORKDIR /app

# Install system dependencies
# build-essential: Required for compiling some Python packages (numpy, scipy)
# curl: Needed for health checks
# gosu: For dropping privileges in entrypoint
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gosu \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
# If requirements don't change, this layer is cached and pip install is skipped
COPY requirements.lock .

# Install Python dependencies from the fully pinned lock file (same resolution
# CI uses). --no-cache-dir: Don't store pip cache (reduces image size)
RUN pip install --no-cache-dir -r requirements.lock

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY tests/ ./tests/
COPY pytest.ini .
COPY serve.py .
COPY train.py .
COPY demo_schema_alignment.py .

# Create user with specified UID/GID to avoid permission issues
RUN groupadd -g ${GID} appuser && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash appuser

# Create directories for models and diagnostics (mounted as volumes)
RUN mkdir -p models diagnostics logs && \
    chown -R ${UID}:${GID} /app

# Expose port 8000 for the API
EXPOSE 8000

# Set environment variables
# PYTHONUNBUFFERED=1: Show print/log statements immediately (critical for debugging)
# PYTHONDONTWRITEBYTECODE=1: Don't create .pyc files (cleaner container)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy and setup entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Health check: Verify service is responding
# Runs every 30 seconds, fails after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint to fix permissions and drop privileges
# Starts as root, fixes ownership, then runs as non-root user
ENTRYPOINT ["/bin/bash", "/usr/local/bin/docker-entrypoint.sh"]

# Default command for the API server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

