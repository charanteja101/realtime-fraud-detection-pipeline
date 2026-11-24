DATA_FILE ?= PS_20174392719_1491204439457_log.csv
TOOLS = docker compose run --rm tools python -m

.PHONY: help up down reset train replay replay-messy report bench-api sample test lint

help:
	@echo "make train         train XGBoost on data/$(DATA_FILE) (replays rows through features.py)"
	@echo "make up            start redpanda, postgres, api, processor"
	@echo "make replay        stream the CSV at RATE events/s (default 2000)"
	@echo "make replay-messy  same, with 2% duplicates and 2% late events"
	@echo "make report        latency / throughput / dedupe / late-event report"
	@echo "make bench-api     load-test the scoring endpoint alone"
	@echo "make sample        write a small synthetic PaySim-shaped CSV (no Kaggle needed)"
	@echo "make reset         stop everything and delete all volumes (fresh run)"

sample:
	$(TOOLS) fraudstream.synth --rows 50000 --out /data/sample.csv

train:
	$(TOOLS) fraudstream.train --data /data/$(DATA_FILE)

up:
	docker compose up -d --build --wait redpanda postgres api processor

down:
	docker compose down

reset:
	docker compose --profile replay --profile tools down -v

replay:
	DATA_FILE=$(DATA_FILE) docker compose run --rm producer

replay-messy:
	DATA_FILE=$(DATA_FILE) DUP_RATE=0.02 LATE_RATE=0.02 docker compose run --rm producer

report:
	$(TOOLS) fraudstream.report

bench-api:
	$(TOOLS) fraudstream.bench_api --url http://api:8000

test:
	pytest

lint:
	ruff check .
