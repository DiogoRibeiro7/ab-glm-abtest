.PHONY: help install lint type test demo

help:
	@echo "Targets: install lint type test demo"

install:
	poetry install

lint:
	poetry run ruff check src tests scripts

type:
	poetry run mypy src

test:
	poetry run pytest -q

demo:
	poetry run python scripts/run_demo.py
