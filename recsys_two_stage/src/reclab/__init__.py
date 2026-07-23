"""reclab — two-stage recommender systems with honest evaluation protocols."""

# OpenBLAS reads its thread count when numpy first loads it, so this only takes effect
# when `reclab` is imported *before* numpy — which is the case for the CLI entry point
# (importing `reclab.main` imports this package first) and for the Docker image, whose
# WORKDIR env sets the same var container-wide. It does NOT bind when numpy is imported
# first (e.g. under pytest, where test modules and plugins import numpy before reclab —
# `implicit`'s ALS then warns that OpenBLAS is multithreaded); that path is left as-is
# because thread count affects only performance/determinism, never a result. `setdefault`
# keeps the "callers can raise it for the full run" contract: an env value already present
# wins. Single-threaded BLAS keeps the dense solves (EASE, the ALS fold-in) deterministic
# and the scaling sweep's fit times comparable rather than machine-dependent on pool size.
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0"
