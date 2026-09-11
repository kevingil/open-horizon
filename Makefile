.PHONY: build dev test lint fmt check py-test py-lint openapi

# Rust control plane -------------------------------------------------------

build:
	cargo build --release

dev:
	RL_STORE_BACKEND=$${RL_STORE_BACKEND:-sqlite} cargo run -p horizon-server -- serve

test:
	cargo test --workspace

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --all -- --check

fmt:
	cargo fmt --all

openapi:
	cargo run -q -p horizon-server -- openapi --out docs/openapi.json

# Python bridge (verifiers envs, trainers, tokenizers) ----------------------

py-test:
	cd python && .venv/bin/python -m pytest

py-lint:
	cd python && .venv/bin/ruff check horizon_bridge tests

check: lint test py-lint py-test
