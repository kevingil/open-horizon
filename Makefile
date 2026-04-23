.PHONY: test lint fmt check dev

test:
	python -m pytest

lint:
	ruff check src/ tests/

fmt:
	ruff check --fix src/ tests/
	ruff format src/ tests/

check: lint test

dev:
	uvicorn rl_stack.interface.api.app:app --reload --host 127.0.0.1 --port 8000
