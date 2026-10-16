#!/bin/bash
# Запускает ephemeral runner в цикле на хосте: каждый контейнер берёт один job и завершается,
# затем поднимается новый со свежим регистрационным токеном.
# Нужны docker и авторизованный gh (права на администрирование репозитория).
# Настройки — в файле runner.env рядом (образец: runner.env.example), в git он не попадает.
# Значения в runner.env — без кавычек и комментариев в конце строки: файл читает и docker.
set -uo pipefail
cd "$(dirname "$0")"

ENV_FILE="${ENV_FILE:-runner.env}"
[ -f "$ENV_FILE" ] || { echo "Нет $ENV_FILE (см. runner.env.example)" >&2; exit 2; }
REPO="$(grep -E '^REPO=' "$ENV_FILE" | tail -1 | cut -d= -f2-)"
[[ "$REPO" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] \
  || { echo "В $ENV_FILE REPO должен быть вида owner/repo без кавычек (сейчас: '$REPO')" >&2; exit 2; }

# Необязательные ограничения ресурсов контейнера (например RUNNER_CPUS=2, RUNNER_MEMORY=4g).
LIMITS=()
[ -n "${RUNNER_CPUS:-}" ] && LIMITS+=(--cpus "$RUNNER_CPUS")
[ -n "${RUNNER_MEMORY:-}" ] && LIMITS+=(--memory "$RUNNER_MEMORY")

docker build -t ai-review-runner . || exit 1

while true; do
  if ! TOKEN="$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token)"; then
    echo "Не удалось получить регистрационный токен (сеть, права gh?), повтор через 60 с" >&2
    sleep 60
    continue
  fi
  docker rm -f ai-review-runner >/dev/null 2>&1 || true  # остаток прерванного запуска
  # Без монтирования $HOME, .env и SSH-ключей; индекс — в именованном томе; без лишних
  # привилегий. Корневая ФС не read-only: runner пишет в _work и _diag.
  if ! docker run --rm --name ai-review-runner \
      --add-host=host.docker.internal:host-gateway \
      --cap-drop=ALL --security-opt no-new-privileges \
      "${LIMITS[@]}" \
      --env-file "$ENV_FILE" \
      -e RUNNER_TOKEN="$TOKEN" \
      -v ai-review-index:/index \
      ai-review-runner; then
    echo "Контейнер завершился с ошибкой, перезапуск через 10 с" >&2
    sleep 10
  fi
done
