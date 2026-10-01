"""Подключение к локальному серверу Ollama для эмбеддингов (rag/, research/chunking_*.py).

OpenAI-совместимый эндпоинт `/v1/embeddings` (design.md изменения
add-rag-indexing-pipeline, решение «Эмбеддинги») — тот же SDK `openai`, что и у
providers/deepseek_client.py/providers/kimi_client.py, но без обязательного ключа:
Ollama его не проверяет, плейсхолдер здесь просто удовлетворяет конструктор
`OpenAI()` (как и в тех двух модулях — см. их докстринги про то, зачем это нужно).

В отличие от DEEPSEEK_API_KEY/KIMI_API_KEY, доступность Ollama НЕ проверяется при
импорте модуля и не завершает процесс: она нужна только офлайн-CLI (rag/cli.py) и
двум командам бота, `/research_chunking_stats` и `/research_chunking_compare`, а не
основному потоку бота — сбой обнаруживается по месту, при первом реальном вызове
embed_texts() (config.py, докстринг EMBEDDINGS_BASE_URL, объясняет, почему здесь
нет и файловой/сетевой проверки при старте, как у MCP_MOEX_DIR).

ВАЖНО: клиент собран с `httpx.Client(trust_env=False)` — НЕ читает
HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/NO_PROXY из окружения процесса. Воспроизведено на
практике: бот запускается с ALL_PROXY, настроенным для DeepSeek/Kimi (внешние API,
которым прокси действительно нужен), и без этой настройки запросы к локальному
Ollama (`http://localhost:11434`, см. EMBEDDINGS_BASE_URL в config.py) молча уходили
через тот же прокси — сам Ollama (проверено логами контейнера) запрос не получал
вовсе, а прокси на них отвечал `502`, что выглядело как сбой Ollama и не лечилось
никаким числом повторов. Ollama в этом проекте всегда локальный сервис по дизайну
(см. докстринг EMBEDDINGS_BASE_URL) — проксировать обращения к нему не нужно
никогда, в отличие от providers/deepseek_client.py/kimi_client.py, которые сами
проксируются штатным httpx-клиентом OpenAI SDK (и должны продолжать это делать).
"""

from __future__ import annotations

import logging
import time

import httpx
from openai import OpenAI

from config import EMBEDDINGS_BASE_URL, EMBEDDINGS_MODEL

logger = logging.getLogger(__name__)

# api_key — Ollama его не проверяет вовсе, значение здесь только для конструктора
# OpenAI(). http_client — см. докстринг модуля про trust_env=False.
embeddings_client = OpenAI(
    api_key="ollama",
    base_url=EMBEDDINGS_BASE_URL,
    http_client=httpx.Client(trust_env=False),
)

# Автоматические повторы при сбое (тот же принцип, что Agent._update_facts()/
# SmartAgent._check_invariants() — см. CLAUDE.md, «Управление контекстом агента»):
# Ollama выгружает модель из памяти после простоя и подгружает её заново при первом
# следующем запросе; это первое обращение после простоя иногда завершается ошибкой
# сервера (в т.ч. 502 — внутренний "ollama serve" не может пока достучаться до ещё не
# поднявшегося процесса "ollama runner", который реально считает эмбеддинг), а не
# просто задержкой. На практике одной короткой паузы иногда мало: воспроизведён
# случай, когда сбой (502) держался дольше ~7 секунд суммарного времени, включая
# собственные повторы SDK openai поверх каждой попытки — поэтому пауз НЕСКОЛЬКО, с
# нарастанием, а не одна фиксированная. Итоговое время ожидания при полностью
# недоступной Ollama растёт соответственно — это осознанный компромисс: команда
# интерактивная, ответ не должен ждать бесконечно, но и однократной попытки
# оказалось недостаточно на практике.
_RETRY_DELAYS_SECONDS = (2.0, 4.0, 6.0)


def embed_texts(texts: list[str], budget_seconds: float | None = None) -> list[list[float]]:
    """Строит эмбеддинги для списка текстов ОДНИМ запросом к EMBEDDINGS_MODEL и
    возвращает векторы в ТОМ ЖЕ порядке, что и `texts` (сортировка по `item.index`
    ответа — не полагаемся на порядок `response.data` как таковой).

    При сбое повторяет запрос с нарастающей паузой между попытками (см.
    _RETRY_DELAYS_SECONDS выше — всего попыток на одну больше числа пауз). Если и
    все повторы не помогли, поднимает исходное исключение OpenAI SDK
    (AuthenticationError, APIConnectionError, APITimeoutError, APIStatusError и
    т.п. — тот же набор, что перехватывает research/_shared.api_error_to_message) —
    его обрабатывает вызывающий код: rag/cli.py печатает ошибку и останавливается,
    не сохраняя частичный индекс; research/chunking_compare.py переводит исключение
    в сообщение на русском тем же хелпером.

    budget_seconds — необязательный предел СУММАРНОГО времени ожидания (попытки плюс
    паузы между ними; design.md изменения add-smart-agent-rag, решение 10) для
    интерактивного пути (поиск по вопросу пользователя в /smart_agent). Без него
    (None) поведение прежнее — так работают индексация и команды исследования, которым
    важнее дождаться результата, чем ответить быстро. С бюджетом: дедлайн считается один
    раз; таймаут каждой попытки — остаток бюджета, а собственные повторы SDK на запрос
    отключены (иначе они складывались бы с нашими повторами); пауза расписания берётся
    только если после неё остаётся время на ещё одну попытку, иначе ожидание
    прекращается и поднимается последнее исключение (обычно APITimeoutError или
    APIConnectionError).
    """
    if not texts:
        return []

    deadline = time.monotonic() + budget_seconds if budget_seconds is not None else None
    max_attempts = len(_RETRY_DELAYS_SECONDS) + 1
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        client = embeddings_client
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            client = embeddings_client.with_options(timeout=remaining, max_retries=0)
        try:
            response = client.embeddings.create(model=EMBEDDINGS_MODEL, input=texts)
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]
        except Exception as exc:  # noqa: BLE001 — повторяем перед тем, как отдать исключение дальше
            last_exc = exc
            if attempt < max_attempts:
                delay = _RETRY_DELAYS_SECONDS[attempt - 1]
                if deadline is not None and deadline - time.monotonic() <= delay:
                    # Пауза съела бы остаток бюджета — на следующую попытку времени не будет.
                    break
                logger.warning(
                    "Сбой запроса эмбеддингов к Ollama (попытка %d/%d): %s. Повтор через %.0f с.",
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

    if last_exc is None:
        raise TimeoutError(f"Бюджет ожидания эмбеддингов ({budget_seconds} с) исчерпан.")
    raise last_exc
