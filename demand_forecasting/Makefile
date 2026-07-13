.PHONY: help setup build backtest promo-lift plot reproduce check-readme test shell clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Run complete setup (build image, run baseline backtest)
	@./setup.sh

build: ## Build the Docker image
	docker compose build

# If the bind-mount source is missing, the Docker daemon creates it root-owned
# and the non-root container user cannot write to it — so create it host-side.
outputs:
	mkdir -p outputs

backtest: outputs ## Backtest one model, e.g. make backtest ARGS="--model lgbm"
	docker compose run --rm demandcast backtest $(ARGS)

promo-lift: outputs ## Fixed-effects promotion-lift estimation (PPML + OLS)
	docker compose run --rm demandcast promo-lift

plot: outputs ## Plot an example 28-day forecast with P10-P90 band (run lgbm backtest first)
	docker compose run --rm demandcast plot-forecast

reproduce: outputs ## Reproduce every number in the README and DATA_NOTES (~50-60 min)
	docker compose run --rm demandcast backtest --model naive
	docker compose run --rm demandcast backtest --model seasonal_naive
	docker compose run --rm demandcast backtest --model lgbm
	docker compose run --rm demandcast backtest --model lgbm --objective poisson
	docker compose run --rm demandcast backtest --model lgbm --objective l2
	docker compose run --rm demandcast backtest --model sarimax
	docker compose run --rm demandcast backtest --model naive --subset sarimax
	docker compose run --rm demandcast backtest --model seasonal_naive --subset sarimax
	docker compose run --rm demandcast backtest --model lgbm --subset sarimax
	docker compose run --rm demandcast backtest --model lgbm --subset sarimax --train-scope global
	docker compose run --rm demandcast promo-lift
	docker compose run --rm demandcast plot-forecast

check-readme: ## Verify README/DATA_NOTES headline numbers against outputs/ (run after make reproduce)
	@python3 scripts/check_readme.py

test: outputs ## Run the test suite
	docker compose run --rm --entrypoint pytest demandcast tests/ -v

shell: outputs ## Open a bash shell in the container
	docker compose run --rm --entrypoint bash demandcast

clean: ## Remove containers
	docker compose down --remove-orphans
