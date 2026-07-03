# Config-Driven ML - Hydra + Pydantic Configuration Layer

A small, config-driven ML training pipeline built to demonstrate a pattern I consider essential for experiment-heavy ML work: Hydra for config composition and CLI overrides, Pydantic for schema validation, joined at a single validated boundary. The ML task itself (regression on scikit-learn's bundled diabetes dataset) is deliberately simple — the configuration layer is the point. Everything runs locally in Docker with no API keys and no downloads.

## Summary

**Domain:** Configuration management for ML experiments  
**Task:** Regression on the diabetes dataset (bundled with scikit-learn, 442 samples)  
**Pattern:** Pydantic models as single source of truth → Hydra ConfigStore → validated CLI  
**Models:** Gradient boosting (HistGradientBoostingRegressor) and Ridge, swappable via a config group  
**Baseline Result:** GBM RMSE 59.1 / R² 0.34 vs Ridge RMSE 63.0 / R² 0.25 (seed 42)  
**Tech Stack:** Hydra, Pydantic v2, scikit-learn, Docker  
**Test Coverage:** 12 tests (schema constraints, Hydra composition, train/evaluate roundtrip)  
**Setup Time:** ~2 minutes

**What the config layer gives you:**
- Every hyperparameter overridable from the CLI: `mlctl train model=ridge model.alpha=5.0`
- Invalid values rejected with readable errors before any training starts
- Named experiment configs that inherit from the base and change only what differs
- Multirun sweeps: `mlctl train -m model=ridge,gbm seed=0,1,2`
- A config snapshot saved with every run, so any run can be re-evaluated from its output directory alone

---

## Quick Start (2 Minutes)

### Prerequisites

- Docker Desktop installed ([Get Docker](https://docs.docker.com/get-docker/))
  - Includes Docker Compose (no separate install needed)
  - Works on Linux, macOS, and Windows
  - **Note:** Requires Docker Compose V2 (`docker compose` command). If you have an older installation with only V1 (`docker-compose` with hyphen), replace `docker compose` with `docker-compose` in the Makefile.
- CPU: any modern machine (training takes under a second)
- RAM: 2 GB
- Disk space: ~1 GB (Docker image)

**Windows users:** Use `.\setup.ps1` for setup (works out of the box, no tools needed). For the `make` commands below, either install Make (`choco install make`) or use Git Bash/WSL2 where it's preinstalled — or run the equivalent `docker compose` commands directly; they're listed in the [Makefile](Makefile).

### One-Command Setup

**For Linux/macOS:**
```bash
# Clone the repository and navigate to project
git clone https://github.com/Medesen/portfolio.git
cd portfolio/config_driven_ml

# Run automated setup (builds image, trains baseline model)
make setup
```

**For Windows (PowerShell):**
```powershell
git clone https://github.com/Medesen/portfolio.git
cd portfolio\config_driven_ml
.\setup.ps1
```

### Try It Out

The interesting part is overriding configuration from the CLI. Every example below is a complete, working command:

```bash
# Train with defaults (GBM, seed 42)
make train

# Swap the model family via the config group, tweak its hyperparameters
make train ARGS="model=ridge model.alpha=5.0"
make train ARGS="model=gbm model.learning_rate=0.05 model.max_iter=500"

# Run a stored experiment config
make train ARGS="--config-name=gbm_tuned"
make train ARGS="--config-name=ridge_strong"

# Sweep both models across three seeds (6 runs)
make sweep

# Re-score a finished run from its config snapshot
make evaluate RUN=outputs/baseline/gbm/seed_42

# Try to break it — validation catches bad values before training starts
make train ARGS="model.max_iter=-5"
# → Invalid configuration:
#   model.gbm.max_iter
#     Input should be greater than 0 [type=greater_than, input_value=-5, ...]
```

Run `make help` to see all available commands.

### Local Alternative (No Docker)

This project has no services and few dependencies, so it also runs directly:

```bash
pip install -e ".[dev]"     # or: uv pip install -e ".[dev]"
mlctl train model=ridge
pytest tests/
```

---

## What This Project Demonstrates

### The Problem This Pattern Solves

Hydra and Pydantic each solve half of the configuration problem. Hydra gives you composition — defaults lists, config groups, CLI overrides, multirun sweeps — but its native structured configs are dataclass-based with weak validation. Pydantic gives you real schemas — constrained fields, discriminated unions, readable errors — but has no composition or override story. Most ML codebases pick one and live with the missing half. Bridging the two is a recognized pattern in the Hydra community (it comes up in Hydra's issue tracker and in several blog posts); this project is my own small, from-scratch implementation of it.

The two libraries meet at a single boundary (`src/mlctl/config_layer.py`, ~100 lines):

1. **Pydantic models are the single source of truth.** Each config schema is defined once, with types, defaults, and constraints. A small converter registers them with Hydra's ConfigStore (required fields become Hydra's `???` marker, defaults are preserved). No duplicated dataclasses, no YAML/schema drift.

2. **Hydra owns composition.** Defaults lists select from config groups, named experiment YAMLs inherit from the base config and change only what differs, and anything can be overridden from the CLI.

3. **Validation happens exactly once, at the boundary.** The `@config_command` decorator composes the config, resolves interpolations, and validates the result against the Pydantic model. Commands receive a typed model — never a raw `DictConfig` — so downstream code gets IDE completion and type checking, and invalid configs fail with a readable message instead of a stack trace mid-training.

The payoff is visible in the schema for model selection:

```python
class RidgeConfig(BaseModel):
    kind: Literal["ridge"] = "ridge"
    alpha: float = Field(default=1.0, gt=0)

class GBMConfig(BaseModel):
    kind: Literal["gbm"] = "gbm"
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    max_iter: int = Field(default=200, gt=0)

ModelConfig = Annotated[Union[RidgeConfig, GBMConfig], Field(discriminator="kind")]
```

A Pydantic discriminated union and a Hydra config group are the same idea approached from two directions: a closed set of alternatives, each with its own schema. Registering both models under the `model` group makes `mlctl train model=ridge` select the variant *and* validate its hyperparameters against the right schema in one motion.

### Key Technical Decisions

**Validation at the boundary, not throughout:** The composed config is validated once, when the command starts. Everything downstream works with typed Pydantic objects, which keeps the domain code (`pipeline.py`) free of Hydra imports and trivially unit-testable.

**Explicit factory over Hydra's `_target_` instantiation:** Hydra can instantiate classes directly from config via `_target_` fields. I deliberately used a plain factory function (`build_model`) instead. At this scale, `_target_` adds indirection and import-time magic without buying anything; at the scale of dozens of swappable components it earns its keep. Knowing where that line sits is part of the demonstration.

**Config snapshots for reproducibility:** Every training run writes its fully resolved config next to the model artifact. The `evaluate` command rebuilds the exact train/test split from the snapshot alone — the roundtrip test asserts that re-evaluation reproduces the training metrics bit-for-bit.

**Deterministic output paths:** Output directories are built by interpolation (`outputs/${experiment_name}/${model.kind}/seed_${seed}`) rather than timestamps, so sweep runs land in predictable places and re-running a config overwrites its own results instead of accumulating clutter.

**Clean CLI errors:** Both failure modes — missing required values and constraint violations — exit with a readable message and status 1, not a traceback. A config CLI is a user interface; it should behave like one.

**Dict-registry command router:** `mlctl <command>` dispatches through a dict, so adding a command is one line and the usage text stays in sync automatically.

---

## Documentation

- **[README.md](README.md)** (this file) - Quick start and overview
- **[Makefile](Makefile)** - Command shortcuts (run `make help`)
- **[src/mlctl/config_layer.py](src/mlctl/config_layer.py)** - The bridge layer, fully commented

---

## Testing

The project includes 12 tests covering the three layers of the config stack:

- **Schema tests** - Discriminated union selects the right model class; constraint violations (`alpha <= 0`, `test_size >= 1`) are rejected; required fields are enforced
- **Hydra integration tests** - Composition via the real config files, config-group overrides, named experiment configs, and both CLI failure modes (validation error, missing required value) using Hydra's compose API
- **Pipeline roundtrip test** - Train produces model + metrics + snapshot; evaluate reconstructs the split from the snapshot and reproduces the training metrics exactly

```bash
# Run all tests in Docker
make test

# Or locally
pytest tests/ -v
```

**Production considerations:** This suite demonstrates the testing patterns for a config layer but is not exhaustive. For production I would add property-based tests for the Pydantic↔ConfigStore converter and CLI-level end-to-end tests via subprocess.

---

## Project Structure

```
config_driven_ml/
├── src/mlctl/
│   ├── config_layer.py         # The bridge: ConfigStore registration + validated boundary
│   ├── config_models.py        # Pydantic schemas (single source of truth)
│   ├── configs/
│   │   ├── train.yaml          # Base config: selects default model group
│   │   ├── evaluate.yaml
│   │   ├── gbm_tuned.yaml      # Named experiment: inherits train, tweaks GBM
│   │   └── ridge_strong.yaml   # Named experiment: overrides the model group
│   ├── pipeline.py             # Domain logic (no Hydra imports)
│   ├── train.py                # Thin CLI entry: @config_command + run_training
│   ├── evaluate.py             # Thin CLI entry: @config_command + run_evaluation
│   └── main.py                 # Dict-registry command router
├── tests/                      # 12 tests across the three layers
├── Dockerfile                  # Non-root container (host-owned outputs)
├── docker-compose.yml
├── Makefile                    # Command shortcuts
├── setup.sh / setup.ps1        # Automated setup scripts
└── pyproject.toml
```

---

## Configuration

The full schema lives in [`config_models.py`](src/mlctl/config_models.py). Everything is overridable from the CLI:

```bash
# Top-level fields
mlctl train experiment_name=my_exp seed=7 test_size=0.25

# Model group selection + nested fields
mlctl train model=ridge model.alpha=0.5
mlctl train model=gbm model.max_depth=3

# Multirun sweeps over any field
mlctl train -m model=ridge,gbm seed=0,1,2
mlctl train -m model=gbm model.learning_rate=0.01,0.05,0.1
```

Stored experiment configs show the inheritance pattern — `ridge_strong.yaml` in its entirety:

```yaml
defaults:
  - train
  - override /model: ridge
  - _self_

experiment_name: ridge_strong

model:
  alpha: 10.0
```

---

## Frequently Asked Questions

### Why both Hydra and Pydantic? Isn't one enough?

They cover different failure modes. With Hydra alone, a typo'd or out-of-range value flows silently into your training code and fails late (or worse, doesn't fail). With Pydantic alone, you get validation but no composition — no config groups, no stored experiments, no `-m` sweeps, and every script grows its own argparse. The bridge is small (~100 lines) and each library does only what it's good at.

### Why such a simple ML task?

Deliberately. The diabetes dataset is bundled with scikit-learn (no download, no license concerns) and trains in under a second, so the config layer — the actual subject — can be exercised instantly. The pattern is the same one I'd use for a multi-day training job; the pipeline behind the config boundary is swappable by design.

### Why not use Hydra's `_target_` auto-instantiation?

See [Key Technical Decisions](#key-technical-decisions). Short version: at two model families, an explicit factory is clearer than string-based class resolution. The decision reverses somewhere around "many components × many variants", and the config layer wouldn't need to change to support it.

### What would this look like at scale?

The boundary pattern is unchanged; what grows around it: config groups per pipeline stage (data, model, trainer, metrics), versioned config schemas with discriminated unions over a `version` field, `_target_`-based instantiation with dependency injection for runtime resources (DB clients, loggers), and sweep outputs collected into experiment tracking (MLflow/W&B) instead of JSON files.

### Does this work on Mac/Windows/Linux?

Yes. Docker Desktop covers all three; see the platform notes in [Quick Start](#quick-start-2-minutes). The local (non-Docker) path works anywhere with Python ≥3.10.

---

## License

This project is part of an ML portfolio. See main repository for license details.

---

## Related Links

- **Hydra Documentation:** https://hydra.cc/
- **Pydantic Documentation:** https://docs.pydantic.dev/
- **Part of ML Portfolio:** [portfolio](../)

---

**Last Updated:** July 2026  
**Docker Support:** Linux, macOS, Windows  
**Total Setup Time:** ~2 minutes
