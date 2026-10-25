#!/usr/bin/env bash
# Приёмка задачи пула: tasks/check.sh T01. Выход 0 — «сделано». Запуск из корня репозитория.
set -u
id="${1:?usage: check.sh Txx}"
cd "$(dirname "$0")/.."
export TELEGRAM_BOT_TOKEN=x DEEPSEEK_API_KEY=x
PY=.venv/bin/python; PYTEST=.venv/bin/pytest; RUFF=.venv/bin/ruff
BASE=exec-loop-baseline
# Начало текущей задачи: раннер задаёт TASK_BASE (коммит перед задачей), по умолчанию — база пула.
TB="${TASK_BASE:-$BASE}"
fail() { echo "НЕ СДЕЛАНО $id: $*"; exit 1; }

# Коммит с изменениями с момента входа в задачу проверяет раннер; здесь — рабочая копия против базы.
changed() { { git diff --name-only "$TB" -- "$@"; git ls-files --others --exclude-standard -- "$@"; } | grep -v "^tasks/\|^\.venv\|^reports/\|^out/"; true; }
no_removed() { # в файле нельзя удалять и менять строки относительно базы (допустимы добавления)
  [ -z "$(git diff "$BASE" -- "$1" | grep '^-[^-]')" ] || fail "из $1 удалены или изменены строки"; }
cov() { # cov <файл src> <минимум %>
  $PYTEST tests -q --cov=src --cov-report=json:/tmp/exec_loop_cov.json >/dev/null 2>&1
  $PY - "$1" "$2" <<'P' || fail "покрытие $1 ниже $2 %"
import json, sys
d = json.load(open("/tmp/exec_loop_cov.json"))["files"][sys.argv[1]]["summary"]["percent_covered"]
print(f"покрытие {sys.argv[1]}: {d:.1f} %"); sys.exit(0 if d >= float(sys.argv[2]) else 1)
P
}
src_untouched() { [ -z "$(changed src)" ] || fail "src/ менять нельзя: $(changed src | head -3 | tr '\n' ' ')"; }
only_file() { [ "$(changed . | sort -u)" = "$1" ] || fail "изменён не только $1: $(changed . | sort -u | tr '\n' ' ')"; }
regression() { # regression <тест-файл> <файл src>: тесты задачи падают на базовой версии src
  [ -n "$(changed tests)" ] || fail "нет изменений в tests/"
  tmp=$(mktemp -d); git archive "$TB" src | tar -x -C "$tmp"
  cp -r tests "$tmp/tests"; rm -rf "$tmp/tests/__pycache__"
  ( cd "$tmp" && ln -s "$OLDPWD/.venv" .venv && .venv/bin/pytest tests/$1 -q -x >/dev/null 2>&1 ) \
    && { rm -rf "$tmp"; fail "новый тест не падает на старом коде"; }
  rm -rf "$tmp"
}

case "$id" in
T01) no_removed tests/test_dev_assistant_rules.py
     $PYTEST tests/test_dev_assistant_rules.py -q >/dev/null 2>&1 || fail "тесты dev_assistant_rules красные" ;;
T02) no_removed tests/test_price_watch_texts.py
     $PYTEST tests/test_price_watch_texts.py -q >/dev/null 2>&1 || fail "тесты price_watch_texts красные" ;;
T03) no_removed tests/test_news_digest_rules.py
     $RUFF check src/news_digest/rules.py >/dev/null 2>&1 || fail "ruff по news_digest/rules.py" ;;
T04) $PY -c "
import sys; sys.path.insert(0,'src')
from news_digest import rules
assert rules.messages_left(9, 6) == 0 and rules.messages_left(2, 6) == 4" || fail "messages_left"
     regression test_news_digest_rules.py src/news_digest/rules.py ;;
T05) $PY -c "
import sys; sys.path.insert(0,'src')
from llm_queue import rules
assert rules.select_by_number(['a','b'], 0) is None and rules.select_by_number(['a','b'], 2) == 'b'" || fail "select_by_number"
     regression test_llm_queue_rules.py src/llm_queue/rules.py ;;
T06) src_untouched; cov src/news_digest/rules.py 100 ;;
T07) src_untouched; cov src/research/_shared.py 100 ;;
T08) src_untouched; cov src/price_watch/scheduler.py 99 ;;
T09) src_untouched; cov src/research/chunking_stats.py 95 ;;
T10) $PY -c "
import sys; sys.path.insert(0,'src')
from llm_queue.rules import format_duration as f
assert [f(0), f(45.4), f(60), f(125), f(-3)] == ['0 с', '45 с', '1 мин', '2 мин 5 с', '0 с'], [f(0), f(45.4), f(60), f(125), f(-3)]" || fail "format_duration"
     [ -n "$(changed tests)" ] || fail "нет тестов" ;;
T11) $PY -c "
import sys; sys.path.insert(0,'src')
from news_digest.rules import count_by_importance as c
items=[{'analysis':{'importance':'high'}},{'analysis':{'importance':'high'}},{'analysis':None},{},{'analysis':{'importance':'??'}},{'analysis':{'importance':'low'}}]
r=c(items)
assert list(r)==['high','medium','low','unanalyzed'] and list(r.values())==[2,0,1,3], r
assert list(c([]).values())==[0,0,0,0]" || fail "count_by_importance"
     [ -n "$(changed tests)" ] || fail "нет тестов" ;;
T12) $PY -c "
import sys; sys.path.insert(0,'src')
from dev_assistant.rules import format_user_ids as f, parse_user_ids as p
assert f([1,2,3])=='1,2,3' and f([])=='' and f([5,5,3,5])=='5,3'
assert p(f([7,7,2]))==(7,2)" || fail "format_user_ids"
     [ -n "$(changed tests)" ] || fail "нет тестов" ;;
T13) $PY -c "
import sys; sys.path.insert(0,'src')
from price_watch import texts
assert texts.every_text('١٥m')=='١٥m' and texts.every_text('15m')=='раз в 15 минут' and texts.every_text('1h')=='раз в час'" || fail "every_text"
     regression test_price_watch_texts.py src/price_watch/texts.py ;;
T14) [ "$(grep -c 'analysis\.get("importance")' src/news_digest/rules.py)" = 1 ] || fail "выражение get(\"importance\") должно быть одно"
     grep -q '_importance' src/news_digest/rules.py || fail "нет помощника _importance"
     $RUFF check src/news_digest/rules.py >/dev/null 2>&1 || fail "ruff"
     no_removed tests/test_news_digest_rules.py ;;
T15) $RUFF check src/rag/cli.py src/rag/extraction.py src/providers/main_client.py >/dev/null 2>&1 || fail "ruff по трём файлам"
     $PY -m compileall -q src >/dev/null 2>&1 || fail "compileall" ;;
T16) only_file src/price_watch/CLAUDE.md
     grep -q 'RECHECK_SECONDS` (час)' src/price_watch/CLAUDE.md || fail "не исправлено значение RECHECK_SECONDS" ;;
T17) only_file src/llm_queue/CLAUDE.md
     grep -q 'окно 20 длительностей' src/llm_queue/CLAUDE.md || fail "не исправлено окно статистики" ;;
T18) only_file src/dev_assistant/CLAUDE.md
     grep -q '`DEV_ASSISTANT_TOP_K` (5)' src/dev_assistant/CLAUDE.md || fail "не исправлен TOP_K" ;;
*) fail "неизвестная задача" ;;
esac

# Подсаженные красные тесты чужих задач T01–T03 ещё могут быть красными: они не считаются провалом.
RED_T01="tests/test_dev_assistant_rules.py::test_invalid_elements_raise[0]"
RED_T02="tests/test_price_watch_texts.py::test_every_text_declines_units"
RED_T03="tests/test_news_digest_rules.py::test_sort_by_importance_then_freshness_with_unanalyzed_last"
deselect=()
for other in T01 T02 T03; do
  [ "$other" = "$id" ] && continue
  var="RED_$other"; deselect+=(--deselect "${!var}")
done
$PYTEST tests -q "${deselect[@]}" >/dev/null 2>&1 || fail "полный pytest красный (кроме чужих подсаженных T01–T03)"
echo "СДЕЛАНО $id"
