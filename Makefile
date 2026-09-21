.PHONY: install run test clean

install:
	pip install -r requirements.txt

run:
	python src/main.py

test:
	@command -v pytest >/dev/null 2>&1 || { \
		echo "pytest не установлен (он не входит в requirements.txt, как и ruff):"; \
		echo "  pip install pytest"; \
		exit 1; \
	}
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
