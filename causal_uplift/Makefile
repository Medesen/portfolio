.PHONY: help setup build balance ate cuped uplift reproduce check-readme test shell clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Run complete setup (build image, run the A/B analysis)
	@./setup.sh

build: ## Build the Docker image
	docker compose build

# If the bind-mount source is missing, the Docker daemon creates it root-owned
# and the non-root container user cannot write to it — so create it host-side.
outputs:
	mkdir -p outputs

balance: outputs ## Covariate-balance check across the randomized arms
	docker compose run --rm upliftlab balance

ate: outputs ## Average treatment effects with honest inference (CIs, MDE, multiplicity)
	docker compose run --rm upliftlab ate

cuped: outputs ## CUPED / regression-adjustment variance reduction
	docker compose run --rm upliftlab cuped

uplift: outputs ## Uplift models (S/T/X-learners), Qini evaluation, targeting simulation
	docker compose run --rm upliftlab uplift $(ARGS)

reproduce: outputs ## Reproduce every number and figure in the README (~5 min)
	docker compose run --rm upliftlab all

check-readme: ## Verify README headline numbers against outputs/ (run after make reproduce)
	@python3 scripts/check_readme.py

test: outputs ## Run the test suite
	docker compose run --rm --entrypoint pytest upliftlab tests/ -v

shell: outputs ## Open a bash shell in the container
	docker compose run --rm --entrypoint bash upliftlab

clean: ## Remove containers
	docker compose down --remove-orphans
