#!/usr/bin/env bash
# QA-прогон обоих уровней: код-тесты и smoke-сценарии. Запускается и в GitHub Actions
# (.github/workflows/qa.yml), и локально (`make qa`) — одна последовательность, один формат
# результатов. Оснастка вне бота: `src/` её не использует.
#
#   scripts/qa_ci.sh [OUT_DIR] [BASE_REF]      (по умолчанию reports/qa-ci и main)
#   PYTHON=.venv/bin/python scripts/qa_ci.sh   (интерпретатор; по умолчанию python)
#
# В OUT_DIR: status.txt (`ключ=код выхода` для compile/ruff/pytest/smoke и счётчики ruff),
# changed.txt, *.txt (вывод проверок), smoke/ (отчёт и скриншоты). Единый комментарий из них
# собирает scripts/qa_report.py. Код выхода: 0 — всё прошло, 1 — есть падения.
#
# ВНИМАНИЕ: скрипт ИСПОЛНЯЕТ код дерева (pytest импортирует src/ и tests/). В CI он запускается
# только на GitHub-hosted runner (одноразовая ВМ без секретов); на self-hosted runner, где работает
# AI-ревью, его запускать нельзя — там код PR только данные.

set -uo pipefail

cd "$(dirname "$0")/.."
OUT=${1:-reports/qa-ci}
BASE=${2:-main}
PY=${PYTHON:-python}

rm -rf "$OUT"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
STATUS="$OUT/status.txt"
: > "$STATUS"

# --- изменённые файлы относительно основной ветки ---------------------------------------------
if ! git diff --name-only --diff-filter=ACMR "$BASE"...HEAD > "$OUT/changed.txt" 2> "$OUT/git.txt"; then
    : > "$OUT/changed.txt"
fi
grep '\.py$' "$OUT/changed.txt" > "$OUT/changed_py.txt" || true

# --- уровень 1: синтаксис ----------------------------------------------------------------------
"$PY" -m compileall -q src > "$OUT/compile.txt" 2>&1
echo "compile=$?" >> "$STATUS"

# --- уровень 1: ruff по изменённым файлам, «до» (основная ветка) и «после» (дерево) -------------
count_ruff() {  # stdin: список файлов; вывод ruff — в файл $1; печатает число замечаний
    local report="$1" rc
    xargs -r "$PY" -m ruff check --output-format concise > "$report" 2>&1
    rc=$?
    RUFF_RC=$rc
    grep -cE '^[^ ]+:[0-9]+:[0-9]+: ' "$report" || true
}

RUFF_RC=0
after=$(count_ruff "$OUT/ruff_after.txt" < "$OUT/changed_py.txt")
after_rc=$RUFF_RC

before=0
tmp=$(mktemp -d)
if git worktree add -q --detach "$tmp" "$BASE" 2> "$OUT/worktree.txt"; then
    before=$(cd "$tmp" && while read -r f; do [ -f "$f" ] && echo "$f"; done < "$OUT/changed_py.txt" \
        | count_ruff "$OUT/ruff_before.txt")
    git worktree remove --force "$tmp" > /dev/null 2>&1 || true
else
    echo "основная ветка недоступна: «до» принято за 0" >> "$OUT/ruff_before.txt"
fi
rm -rf "$tmp"

# Провал — рост числа замечаний в изменённых файлах или сбой самого ruff (код 2): старые
# замечания (в основном E501) PR не обязан чинить, но и добавлять новые не должен.
ruff_status=0
if [ "$after" -gt "$before" ] || [ "$after_rc" -ge 2 ]; then
    ruff_status=1
fi
{
    echo "ruff=$ruff_status"
    echo "ruff_before=$before"
    echo "ruff_after=$after"
} >> "$STATUS"

# --- уровень 1: тесты и покрытие ---------------------------------------------------------------
TELEGRAM_BOT_TOKEN=x DEEPSEEK_API_KEY=x "$PY" -m pytest tests -q -p no:cacheprovider \
    --cov=src --cov=scripts --cov-report=term > "$OUT/pytest.txt" 2>&1
echo "pytest=$?" >> "$STATUS"

# --- уровень 2: smoke со скриншотами -----------------------------------------------------------
"$PY" scripts/smoke.py --out "$OUT/smoke" > "$OUT/smoke.txt" 2>&1
echo "smoke=$?" >> "$STATUS"
# HTML-страница шага нужна только как запасной вариант, если PNG не получился.
find "$OUT/smoke" -name '*.html' 2> /dev/null | while read -r page; do
    [ -f "${page%.html}.png" ] && rm -f "$page"
done

if grep -qE '^(compile|ruff|pytest|smoke)=[1-9]' "$STATUS"; then
    exit 1
fi
exit 0
