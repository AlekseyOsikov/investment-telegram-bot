.PHONY: install run test clean index tune tune-report

install:
	pip install -r requirements.txt

run:
	python src/main.py

index:
	PYTHONPATH=src python -m rag.cli

# Подбор параметров локальной модели Ollama (scripts/ollama_tune.py): нужен Ollama, индекс и файл
# вариантов в data/ollama_tune/. Аргументы скрипта — через ARGS, например:
#   make tune ARGS="--no-judge --repeats 1"      make tune-report ARGS=3
tune:
	python scripts/ollama_tune.py run $(ARGS)

tune-report:
	python scripts/ollama_tune.py report $(ARGS)

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
