#!/bin/bash
# Регистрирует ephemeral runner (один job — один контейнер) и запускает его.
set -euo pipefail

: "${REPO:?не задан REPO (owner/repo)}"
: "${RUNNER_TOKEN:?не задан RUNNER_TOKEN (регистрационный токен)}"

./config.sh --unattended --ephemeral --replace \
  --url "https://github.com/${REPO}" \
  --token "${RUNNER_TOKEN}" \
  --labels ai-review \
  --name "ai-review-$(hostname)"

# Токен регистрации больше не нужен: из окружения процессов job он убирается. В метаданных
# контейнера (`docker inspect`) он остаётся, но одноразовый и живёт час.
unset RUNNER_TOKEN
exec ./run.sh
