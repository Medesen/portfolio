FROM python:3.12-slim

# implicit (ALS) uses OpenMP at runtime; slim images don't ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root so files written to the mounted outputs/ volume are owned
# by the host user (UID 1000 is the typical first user on Linux).
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app

WORKDIR /app

# Dependencies come from requirements.lock (pip-compile of pyproject.toml, incl.
# the dev extra) so rebuilds are reproducible; installed before the source copy
# so code changes don't invalidate the dependency layer. The PyTorch CPU wheel
# index keeps the image from pulling ~2 GB of unused CUDA libraries — the neural
# models (Stage 2) are CPU-only by design.
COPY requirements.lock ./
# hnswlib ships as a C++ extension with no wheel for this Python, so it compiles
# from source. Install the toolchain, build, and purge it in the same layer so the
# runtime image does not carry a compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.lock \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY data ./data
COPY tests ./tests

USER app

ENTRYPOINT ["reclab"]
