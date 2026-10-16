#!/bin/bash
# Запускает ephemeral runner в цикле на хосте: каждый контейнер берёт один job и завершается,
# затем поднимается новый со свежим регистрационным токеном.
# Нужны docker и авторизованный gh (права на администрирование репозитория).
# Настройки — в файле runner.env рядом (образец: runner.env.example), в git он не попадает.
set -euo pipefail
cd "$(dirname "$0")"

ENV_FILE="${ENV_FILE:-runner.env}"
[ -f "$ENV_FILE" ] || { echo "Нет $ENV_FILE (см. runner.env.example)" >&2; exit 2; }
REPO="$(grep -E '^REPO=' "$ENV_FILE" | cut -d= -f2-)"
[ -n "$REPO" ] || { echo "В $ENV_FILE не задан REPO" >&2; exit 2; }

docker build -t ai-review-runner .

while true; do
  TOKEN="$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token)"
  # Без монтирования $HOME, .env и SSH-ключей; индекс — в именованном томе.
  docker run --rm --name ai-review-runner \
    --add-host=host.docker.internal:host-gateway \
    --env-file "$ENV_FILE" \
    -e RUNNER_TOKEN="$TOKEN" \
    -v ai-review-index:/index \
    ai-review-runner || echo "Контейнер завершился с ошибкой, перезапуск через 10 с" >&2
  sleep 10
done
