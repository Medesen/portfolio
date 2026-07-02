.PHONY: help setup build train evaluate sweep test shell clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Run complete setup (build image, train baseline model)
	@./setup.sh

build: ## Build the Docker image
	docker compose build

# If the bind-mount source is missing, the Docker daemon creates it root-owned
# and the non-root container user cannot write to it — so create it host-side.
outputs:
	mkdir -p outputs

train: outputs ## Train a model. Pass Hydra overrides via ARGS, e.g. make train ARGS="model=ridge seed=7"
	docker compose run --rm mlctl train $(ARGS)

evaluate: outputs ## Re-score a run, e.g. make evaluate RUN=outputs/baseline/gbm/seed_42
	docker compose run --rm mlctl evaluate run_dir=$(RUN)

sweep: outputs ## Multirun example: both models x three seeds
	docker compose run --rm mlctl train -m model=ridge,gbm seed=0,1,2 experiment_name=sweep

test: outputs ## Run the test suite
	docker compose run --rm --entrypoint pytest mlctl tests/ -v

shell: outputs ## Open a bash shell in the container
	docker compose run --rm --entrypoint bash mlctl

clean: ## Remove containers
	docker compose down --remove-orphans
