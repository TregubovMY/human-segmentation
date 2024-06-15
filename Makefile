.PHONY: dependencies env clean docs shell data_proc train valid visual models

env:
	@echo "Activating virtual environment..."
	python -m venv env

dependencies: env
	@echo "Installing dependencies..."
	.\env\Scripts\activate
	pip install -r requirements.txt

shell:
	.\env\Scripts\activate
	
clean:
	$(MAKE) -C docs clean
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.log" -delete

docs:
	$(MAKE) -C docs livehtml

data_proc:
	python -m src.models.deeplabv3plus

train:
	python -m src.trainer.train

valid:
	python -m src.validations.validation

visual:
	python -m src.visualization.visualization_main

models:
	python ./models/load_models.py

data:
	python ./data/load_data.py --source people_segmentation
