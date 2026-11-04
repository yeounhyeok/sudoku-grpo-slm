.PHONY: test data models models-all
test:
	python tests/test_grid.py
data:
	python scripts/prepare_data.py
models:
	python scripts/download_models.py
models-all:
	python scripts/download_models.py --all
