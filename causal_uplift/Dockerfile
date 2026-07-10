FROM python:3.12-slim

# LightGBM needs OpenMP at runtime; slim images don't ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root so files written to the mounted outputs/ volume are owned
# by the host user (UID 1000 is the typical first user on Linux).
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dev]"

COPY data ./data
COPY tests ./tests

USER app

ENTRYPOINT ["upliftlab"]
