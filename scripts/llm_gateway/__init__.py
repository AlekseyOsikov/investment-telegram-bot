"""LLM Gateway: HTTP-прокси между пользователем и LLM с input/output guard.

Одноразовая оснастка курса по security (как scripts/attack.py и
scripts/indirect_injection.py): `src/` её не импортирует, поведение бота не меняет.

Публичный API:
    create_app(settings=None, provider=None) -> FastAPI
    Settings.from_env()

Запуск сервера:
    python -m llm_gateway --host 127.0.0.1 --port 8000   (из каталога scripts/)
"""

from .app import create_app
from .settings import Settings

__all__ = ["create_app", "Settings"]
