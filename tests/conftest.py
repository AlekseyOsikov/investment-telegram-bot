"""Общая настройка pytest: делает пакеты из src/ импортируемыми.

Отдельного пакета/установки у проекта нет (запуск — `python src/main.py`), поэтому
путь добавляется здесь, а не через setup.py или PYTHONPATH в Makefile.
"""

import os
import sys
from pathlib import Path

# Тестам сеть не нужна, а часть модулей src/ создаёт HTTP-клиентов при импорте: схему `socks://`
# из окружения разработчика httpx не понимает (нужна `socks5://`) и роняет импорт. Прокси снимается
# для всей сессии pytest здесь, а не копиями в отдельных тест-файлах; scripts/smoke.py делает то же.
for _name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_name, None)
    os.environ.pop(_name.lower(), None)

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Чистые правила оснастки scripts/ (ollama_tune_lib.py): скрипты не пакет, как и src/.
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Чистые правила оснастки «локальная модель как код-ассистент» (scripts/ollama/local_llm_lib.py):
# скрипты запускаются файлами, поэтому каталог добавляется отдельно, как у mcp_servers ниже.
SCRIPTS_OLLAMA = SCRIPTS / "ollama"
if str(SCRIPTS_OLLAMA) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_OLLAMA))

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

# Сервер новостей (mcp_servers/news/news_*.py): как и project_files, не пакет и `src/` его не
# импортирует; модули названы с префиксом `news_`, чтобы не столкнуться с `server`/`rules` других.
MCP_NEWS = Path(__file__).resolve().parent.parent / "mcp_servers" / "news"
if str(MCP_NEWS) not in sys.path:
    sys.path.insert(0, str(MCP_NEWS))
