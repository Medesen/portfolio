# Customer Churn Prediction API - Dockerfile
# Two-stage build: compilers and git exist only in the builder stage, so the
# runtime image ships neither build tooling nor its CVE surface.
#
# Deliberate trade-off: the test suite (tests/, pytest, locust) IS part of the
# runtime image — this project's documented workflow is Docker-only testing
# ("make test" runs pytest inside the same image that serves), and the load
# test runs locust from the api image. A stricter deployment would split a
# separate requirements-runtime.lock; here the single lock keeps CI, tests,
# and serving on one pinned resolution.

# ============================ builder ============================
FROM python:3.12-slim AS builder

# build-essential/git: only needed while resolving and building wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock .

# Install the fully pinned lock file (same resolution CI uses) into a venv
# that can be copied wholesale into the runtime stage
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.lock

# ============================ runtime ============================
FROM python:3.12-slim

# Accept UID/GID as build arguments (defaults to 1000:1000)
ARG UID=1000
ARG GID=1000
ARG NB_USER=appuser

# Make IDs available to the entrypoint at runtime
ENV NB_USER=${NB_USER} NB_UID=${UID} NB_GID=${GID}

# Set working directory in container
WORKDIR /app

# Runtime system dependencies only
# curl: Needed for health checks
# gosu: For dropping privileges in entrypoint
RUN apt-get update && apt-get install -y \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies come pre-built from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

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

