"""Подключение к локальному серверу Ollama для ЧАТ-вызовов (переписывание вопроса,
providers/rewrite_client.py; design.md изменения add-rag-rerank-and-rewrite, решение 2).

OpenAI-совместимый эндпоинт `/v1/chat/completions` — тот же SDK `openai`, что у
providers/deepseek_client.py и providers/kimi_client.py, но без ключа: Ollama его не
проверяет, плейсхолдер лишь удовлетворяет конструктор `OpenAI()` (см. докстринг
providers/embeddings_client.py).

Клиент НЕ связан ни с MAIN_CLIENT (основной поток), ни с embeddings_client.py: адрес чат-сервера
задаёт OLLAMA_BASE_URL, адрес эмбеддингов — EMBEDDINGS_BASE_URL, оператор может держать их на
разных серверах. Доступность сервера при импорте не проверяется и процесс не завершает:
сбой обнаруживается по месту, при вызове, и обрабатывается откатом к исходному вопросу.

Клиент собран с `httpx.Client(trust_env=False)` — не читает HTTP_PROXY/HTTPS_PROXY/ALL_PROXY из
окружения: Ollama локальный сервис, а прокси, настроенный для DeepSeek/Kimi, иначе молча
перехватывал бы запросы к нему (подробности — в докстринге embeddings_client.py).
"""

from __future__ import annotations

import httpx
from openai import OpenAI

from config import OLLAMA_BASE_URL

# api_key — Ollama его не проверяет, значение только для конструктора OpenAI().
ollama_client = OpenAI(
    api_key="ollama",
    base_url=OLLAMA_BASE_URL,
    http_client=httpx.Client(trust_env=False),
)
