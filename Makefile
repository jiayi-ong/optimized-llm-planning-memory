.PHONY: install lint typecheck test train eval episode clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

# Run a single debug episode (raw trajectory mode, no compression)
episode:
	python scripts/run_episode.py agent.mode=raw

# Run PPO training with default config
train:
	python scripts/run_training.py

# Run evaluation on the test set
eval:
	python scripts/run_evaluation.py agent.mode=compressor

# Generate diverse user request variations from template
generate-requests:
	python scripts/generate_user_requests.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
