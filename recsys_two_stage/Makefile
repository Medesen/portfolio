.PHONY: help setup build eda tune evaluate sampled protocols beyond reproduce check-readme test shell clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup: ## Run complete setup (build image, run a quick evaluation)
	@./setup.sh

build: ## Build the Docker image
	docker compose build

# If the bind-mount source is missing, the Docker daemon creates it root-owned
# and the non-root container user cannot write to it — so create it host-side.
outputs:
	mkdir -p outputs

eda: outputs ## Dataset summary + filter-threshold grid
	docker compose run --rm reclab eda

tune: outputs ## Grid search on the validation window (slowest; regenerates TUNED_PARAMS)
	docker compose run --rm reclab tune

evaluate: outputs ## Full-catalogue evaluation of all models (the honest protocol)
	docker compose run --rm reclab evaluate

sampled: outputs ## Sampled-negative evaluation + the disagreement table
	docker compose run --rm reclab sampled

protocols: outputs ## Temporal vs leave-one-out comparison
	docker compose run --rm reclab protocols

beyond: outputs ## Coverage / Gini / popularity-bias metrics
	docker compose run --rm reclab beyond

features: outputs ## Item content-feature coverage (as-of cutoff)
	docker compose run --rm reclab features

neural: outputs ## Train + evaluate the two-tower and SASRec models (~14 min, CPU)
	docker compose run --rm reclab neural

ablations: outputs ## logQ on/off, id vs content, full vs sampled loss
	docker compose run --rm reclab ablations

ceiling: outputs ## Retrieval-ceiling analysis + blending + figure
	docker compose run --rm reclab ceiling

cold-start: outputs ## Cold-item evaluation vs the category-popularity baseline
	docker compose run --rm reclab cold-start

reproduce: outputs ## Reproduce every number in the README (Stage 1 + Stage 2, ~30 min)
	docker compose run --rm reclab all

check-readme: ## Verify README headline numbers against outputs/ (run after reproduce)
	@python3 scripts/check_readme.py

test: outputs ## Run the test suite
	docker compose run --rm --entrypoint pytest reclab tests/ -v

shell: outputs ## Open a bash shell in the container
	docker compose run --rm --entrypoint bash reclab

clean: ## Remove containers
	docker compose down --remove-orphans
