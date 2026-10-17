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

# Чистая выборка мок-CRM поддержки (mock_servers/support_crm/crm_store.py): сервер не пакет и
# `src/` его не импортирует; тесты берут только чистый модуль.
MOCK_CRM = Path(__file__).resolve().parent.parent / "mock_servers" / "support_crm"
if str(MOCK_CRM) not in sys.path:
    sys.path.insert(0, str(MOCK_CRM))

# Чистые правила MCP-сервера файлов проекта (mcp_servers/project_files/files_rules.py): сервер не
# пакет и `src/` его не импортирует; тесты берут модули сервера напрямую.
MCP_PROJECT_FILES = Path(__file__).resolve().parent.parent / "mcp_servers" / "project_files"
if str(MCP_PROJECT_FILES) not in sys.path:
    sys.path.insert(0, str(MCP_PROJECT_FILES))
