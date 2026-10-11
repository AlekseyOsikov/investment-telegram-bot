"""Общая настройка pytest: делает пакеты из src/ импортируемыми.

Отдельного пакета/установки у проекта нет (запуск — `python src/main.py`), поэтому
путь добавляется здесь, а не через setup.py или PYTHONPATH в Makefile.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Чистые правила оснастки scripts/ (ollama_tune_lib.py): скрипты не пакет, как и src/.
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
