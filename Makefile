.PHONY: dependencies env clean docs

dependencies:
	@echo "Installing dependencies..."
	poetry install
	@echo "Initializing pre-commit..."

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell

clean:
	$(MAKE) -C docs clean
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.log" -delete

docs:
	$(MAKE) -C docs livehtml
