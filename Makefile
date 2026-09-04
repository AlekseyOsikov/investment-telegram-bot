.PHONY: install run test clean

install:
	pip install -r requirements.txt

run:
	python src/main.py

test:
	@echo "Тестов пока нет. Заглушка для будущего pytest-сьюта."
	@echo "Когда появятся тесты: pytest tests/"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
