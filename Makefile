.PHONY: dependencies env clean docs

dependencies:
	@echo "Initializing Git..."
	git init
	@echo "Installing dependencies..."
	poetry install
	@echo "Initializing pre-commit..."

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell

dvc:
	@echo "Initializing dvc..."
	dvc init

clean:
	$(MAKE) -C docs clean
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

docs:
	$(MAKE) -C docs livehtml