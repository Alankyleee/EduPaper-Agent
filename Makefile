.PHONY: install run test lint eval docker

install:
	python -m pip install -e ".[dev]"

run:
	python start.py

test:
	pytest

lint:
	ruff check .

eval:
	python eval/run_eval.py

docker:
	docker compose up --build
