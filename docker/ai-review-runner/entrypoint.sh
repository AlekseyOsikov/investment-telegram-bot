#!/bin/bash
# Регистрирует ephemeral runner (один job — один контейнер) и запускает его.
set -euo pipefail

: "${REPO:?не задан REPO (owner/repo)}"
: "${RUNNER_TOKEN:?не задан RUNNER_TOKEN (регистрационный токен)}"

./config.sh --unattended --ephemeral \
  --url "https://github.com/${REPO}" \
  --token "${RUNNER_TOKEN}" \
  --labels ai-review \
  --name "ai-review-$(hostname)"

# Токен регистрации больше не нужен и не должен остаться в окружении job.
unset RUNNER_TOKEN
exec ./run.sh
