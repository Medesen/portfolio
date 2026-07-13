# Dockerfile
FROM python:3.11-slim

# System deps (gosu for dropping privileges at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gosu \
  && rm -rf /var/lib/apt/lists/*

# ----- Create a real non-root user with configurable UID/GID -----
ARG NB_USER=app
ARG NB_UID=1000
ARG NB_GID=1000
RUN groupadd -g ${NB_GID} ${NB_USER} \
 && useradd -m -u ${NB_UID} -g ${NB_GID} -s /bin/bash ${NB_USER}

# Make IDs available to the entrypoint at runtime
ENV NB_USER=${NB_USER} NB_UID=${NB_UID} NB_GID=${NB_GID}

# ----- App setup -----
WORKDIR /app
# Install from the pinned lock for reproducible builds (requirements.txt holds
# the human-readable intent and is copied in with the rest of the source below).
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY . .

# Ensure project dirs exist and are owned by the non-root user
RUN mkdir -p /app/data/processed /app/data/state /app/data/vector_store /app/logs \
 && chown -R ${NB_UID}:${NB_GID} /app

# Modern cache locations (avoid TRANSFORMERS_CACHE deprecation warning)
ENV HF_HOME=/home/${NB_USER}/.cache/huggingface \
    TORCH_HOME=/home/${NB_USER}/.cache/torch \
    PYTHONUNBUFFERED=1

# ----- Entrypoint: normalize line endings, chmod, and exec via bash -----
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Call through bash to avoid shebang/format issues entirely
ENTRYPOINT ["/bin/bash", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]
