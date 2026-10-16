.PHONY: install run test clean index dev-index review-index tune tune-report

install:
	pip install -r requirements.txt

run:
	python src/main.py

index:
	PYTHONPATH=src python -m rag.cli

# Индекс документации проекта для режима /dev (README, CLAUDE.md, openspec/specs): отдельный
# каталог DEV_ASSISTANT_INDEX_DIR, нужен локальный Ollama с EMBEDDINGS_MODEL.
dev-index:
	PYTHONPATH=src python -m dev_assistant.cli

# Индекс AI-ревью PR (документация + исходники src/ и scripts/) в PR_REVIEW_INDEX_DIR
# (по умолчанию data/pr_review_index): нужен локальный Ollama с EMBEDDINGS_MODEL.
review-index:
	python scripts/pr_review.py index

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
