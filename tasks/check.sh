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
T06) src_untouched; cov src/research/chunking_compare.py 57 ;;
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
     line=$(grep 'RECHECK_SECONDS' src/price_watch/CLAUDE.md | head -1)
     echo "$line" | grep -Eq '(^|[^0-9])(10 мин|600 с)' && fail "RECHECK_SECONDS в тексте всё ещё не равен 3600 с"
     echo "$line" | grep -Eq 'час|3600|60 мин' || fail "в тексте нет значения RECHECK_SECONDS (3600 с = час)" ;;
T17) only_file src/llm_queue/CLAUDE.md
     grep -q 'окно 50' src/llm_queue/CLAUDE.md && fail "окно статистики в тексте всё ещё 50"
     grep -Eq 'окно[^.;]{0,20}\b20\b|\b20\b[^.;]{0,20}длительност' src/llm_queue/CLAUDE.md || fail "в тексте нет окна 20" ;;
T18) only_file src/dev_assistant/CLAUDE.md
     grep -q 'DEV_ASSISTANT_TOP_K` (10)' src/dev_assistant/CLAUDE.md && fail "TOP_K в тексте всё ещё 10"
     grep -Eq 'DEV_ASSISTANT_TOP_K`[^,]{0,10}\b5\b' src/dev_assistant/CLAUDE.md || fail "в тексте нет TOP_K = 5" ;;
T19) $PY -c "
import sys; sys.path.insert(0,'src')
from price_watch.scheduler import split_message
long = ''.join(chr(0x410 + i % 32) for i in range(9000))
parts = split_message(long, 4000)
assert ''.join(parts) == long and all(len(p) <= 4000 for p in parts), [len(p) for p in parts]
parts = split_message('коротко\n' + long, 4000)
assert parts[0] == 'коротко' and ''.join(parts[1:]) == long" || fail "split_message теряет символы"
     regression test_price_watch_scheduler.py src/price_watch/scheduler.py ;;
T20) no_removed tests/test_news_digest_analyzer.py
     $PYTEST tests/test_news_digest_analyzer.py -q >/dev/null 2>&1 || fail "тест news_digest_analyzer красный"
     regression test_llm_queue_gate.py src/llm_queue/gate.py ;;
T21) for f in __init__.py texts.py commands.py; do [ -f src/about/$f ] || fail "нет src/about/$f"; done
     [ -f src/about/CLAUDE.md ] || fail "нет src/about/CLAUDE.md"
     head -c 300 src/about/texts.py | grep -q '"""' || fail "у texts.py нет докстринга модуля"
     grep -q 'from __future__ import annotations' src/about/texts.py src/about/commands.py || fail "нет from __future__ import annotations"
     grep -q 'ChatType.PRIVATE' src/about/commands.py || fail "обработчик /about без фильтра личного чата"
     [ "$(grep -c 'build_about_handler' src/main.py)" -ge 2 ] || fail "build_about_handler не импортирован и не добавлен в main.py"
     grep -q '/about' src/main.py || fail "/about нет в /help"
     [ "$(grep -c '/about' README.md)" -ge 1 ] && grep -q 'about/' README.md || fail "README: нет /about в таблице команд и about/ в дереве"
     grep -q 'about/' CLAUDE.md || fail "карта src/ в CLAUDE.md не знает about/"
     ls tests/test_about_*.py >/dev/null 2>&1 || fail "нет tests/test_about_*.py"
     $PY -c "
import re, sys; sys.path.insert(0,'src')
from about import texts
t = texts.about_text()
assert isinstance(t, str) and len(t) > 100, 'короткий текст'
assert 'не лицензированный финансовый советник' in t.lower(), 'нет оговорки'
bad = re.compile(r'гарантир\w+|точно\s+(?:вырастет|упадёт)|наверняка|рекомендуем\s+купить', re.I)
assert not bad.search(t), bad.search(t).group(0)" || fail "about_text: оговорка или запретные формулировки"
     $RUFF check src/about >/dev/null 2>&1 || fail "ruff по src/about"
     tmp=$(mktemp -d); git archive "$TB" src pyproject.toml | tar -x -C "$tmp"
     before=$(cd "$tmp" && "$OLDPWD/$RUFF" check src/main.py --output-format concise 2>/dev/null | grep -c ': [A-Z][0-9]'); rm -rf "$tmp"
     after=$($RUFF check src/main.py --output-format concise 2>/dev/null | grep -c ': [A-Z][0-9]')
     [ "$after" -le "$before" ] || fail "ruff по main.py: было $before, стало $after" ;;
T22) $PY -c "
import os, subprocess, sys
env = {**os.environ, 'TELEGRAM_BOT_TOKEN': 'x', 'DEEPSEEK_API_KEY': 'x', 'PYTHONPATH': 'src'}
def run(code, **extra):
    return subprocess.run([sys.executable, '-c', code], env={**env, **extra}, capture_output=True, text=True)
assert run('import config', LLM_STATS_WINDOW='0').returncode != 0, 'ноль должен останавливать запуск'
assert run('import config', LLM_STATS_WINDOW='abc').returncode != 0, 'мусор должен останавливать запуск'
r = run('import config; print(config.LLM_STATS_WINDOW)', LLM_STATS_WINDOW='7'); assert r.stdout.strip() == '7', r.stderr[-300:]
r = run('import config; print(config.LLM_STATS_WINDOW)'); assert r.stdout.strip() == '20', r.stderr[-300:]
" || fail "конфигурация LLM_STATS_WINDOW"
     $PY -c "
import sys; sys.path.insert(0,'src')
from llm_queue.gate import LlmGate
g = LlmGate(concurrency=1, max_pending_per_user=1, queue_max=5, ttl_seconds=60, timeout_seconds=60, stats_window=3)
for i in range(6): g.stats.record('main', float(i))
assert g.stats.count('main') == 3" || fail "LlmGate(stats_window=...)"
     grep -q 'LLM_STATS_WINDOW' .env.example || fail "нет LLM_STATS_WINDOW в .env.example"
     grep -q 'LLM_STATS_WINDOW' src/llm_queue/CLAUDE.md || fail "нет LLM_STATS_WINDOW в src/llm_queue/CLAUDE.md"
     grep -q 'LLM_STATS_WINDOW' src/llm_queue/commands.py || fail "start_gate не передаёт настройку шлюзу"
     grep -Eq '^\s*(import config|from config import)' src/llm_queue/rules.py && fail "llm_queue/rules.py не должен импортировать config"
     [ -n "$(changed tests)" ] || fail "нет тестов" ;;
T23) tmp=$(mktemp -d); git archive "$TB" src/price_watch/texts.py | tar -x -C "$tmp"
     $PY - "$tmp/src" <<'P' || { rm -rf "$tmp"; fail "watch_set_text"; }
import re, sys
sys.path.insert(0, "src")
from price_watch import texts
result = {"chat_id": 1, "secids": ["SBER"], "poll_interval": "15m", "report_interval": "1h",
          "replaced": False, "next_poll_at": "2026-09-26T12:15:00+03:00",
          "next_report_at": "2026-09-26T13:00:00+03:00", "first_samples": []}
new = texts.watch_set_text(result)
sys.path.insert(0, sys.argv[1] + "/price_watch")
import importlib.util
spec = importlib.util.spec_from_file_location("old_texts", sys.argv[1] + "/price_watch/texts.py")
old_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_mod)
old = old_mod.watch_set_text(result)
assert texts.DELAY_NOTE in new, "потеряна оговорка о задержке котировок"
for line in old.split("\n"):
    assert line in new, f"пропала прежняя строка: {line!r}"
added = [l for l in new.split("\n") if l not in old.split("\n") and l.strip()]
assert added, "новой строки нет"
bad = re.compile(r"гарантир\w+|точно\s+(?:вырастет|упадёт|узна)|наверняка|раньше\s+других|не\s+пропустите", re.I)
assert not bad.search(new), bad.search(new).group(0)
assert re.search(r"не\s+(?:является\s+)?(?:инвестиционн\w+\s+)?рекомендаци|не\s+гаранти|не\s+лицензированн", new, re.I), "нет оговорки, что это не рекомендация/гарантия"
P
     rm -rf "$tmp"; [ -n "$(changed tests)" ] || fail "нет тестов" ;;
T24) src_untouched; [ -f tests/test_llm_queue_commands.py ] || fail "нет tests/test_llm_queue_commands.py"
     cov src/llm_queue/commands.py 70 ;;
T25) f=docs/research-queue-kinds.md; [ -f "$f" ] || fail "нет $f"
     only_file "$f"
     $PY - "$f" <<'P' || fail "в $f неверные или недостающие ссылки"
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
refs = re.findall(r"([\w./-]+\.py):(\d+)", text)
assert refs, "нет ссылок путь:строка"
good = {}
for path, line in refs:
    try:
        lines = open(path, encoding="utf-8").read().split("\n")
    except OSError:
        raise AssertionError(f"нет файла {path}")
    assert 1 <= int(line) <= len(lines), f"{path}:{line} вне файла"
    src = " ".join(lines[max(int(line) - 3, 0): int(line) + 2])
    for kind in ("KIND_MAIN", "KIND_AGENT", "KIND_COMPARE", "KIND_SMART_AGENT", "KIND_DEV_ASSISTANT",
                 "KIND_SUPPORT", "KIND_RESEARCH", "KIND_NEWS"):
        if kind in src:
            good.setdefault(kind, set()).add((path, line))
missing = {"KIND_MAIN", "KIND_AGENT", "KIND_COMPARE", "KIND_SMART_AGENT", "KIND_DEV_ASSISTANT",
           "KIND_SUPPORT", "KIND_RESEARCH", "KIND_NEWS"} - set(good)
assert not missing, f"нет верной ссылки для {sorted(missing)}"
assert "slot" in text and "news_digest" in text, "не сказано, что разбор новостей идёт через gate.slot, а не enqueue"
P
     ;;
T26) [ -f src/telegram_text.py ] || fail "нет src/telegram_text.py"
     grep -Eq '^\s*(import config|from config import)' src/telegram_text.py && fail "telegram_text.py не должен импортировать config"
     grep -q 'from __future__ import annotations' src/telegram_text.py || fail "нет from __future__ import annotations"
     grep -q 'telegram_text' src/price_watch/scheduler.py || fail "price_watch не использует telegram_text"
     grep -q 'telegram_text' src/dev_assistant/command.py || fail "dev_assistant не использует telegram_text"
     grep -Eq '^\s*(import price_watch|from price_watch)' src/dev_assistant/command.py && fail "dev_assistant импортирует price_watch"
     [ "$(grep -c 'def split_message' src/price_watch/scheduler.py)" = 0 ] || [ "$(grep -c 'telegram_text' src/price_watch/scheduler.py)" -ge 1 ] || fail "дубль"
     $PY -c "
import sys; sys.path.insert(0,'src')
from telegram_text import split_message as f
text = '\n'.join(f'строка {i}' for i in range(2000))
parts = f(text, 100)
assert all(len(p) <= 100 for p in parts) and '\n'.join(parts) == text
assert f('x' * 250, 100) == ['x' * 100, 'x' * 100, 'x' * 50]
from dev_assistant.command import split_for_telegram as g
parts = g(text, 4000)
assert all(len(p) <= 4000 for p in parts) and '\n'.join(parts) == text, 'dev_assistant режет не по строкам'
from price_watch.scheduler import split_message as h
assert h(text, 100) == parts[:0] + f(text, 100)" || fail "поведение нарезки"
     [ -n "$(changed tests)" ] || fail "нет тестов"
     files=$(changed src | grep '\.py$' | tr '\n' ' ')
     tmp=$(mktemp -d); git archive "$TB" src pyproject.toml | tar -x -C "$tmp"
     before=$(cd "$tmp" && for f in $files; do [ -f "$f" ] && "$OLDPWD/$RUFF" check "$f" --output-format concise 2>/dev/null; done | grep -c ': [A-Z][0-9]'); rm -rf "$tmp"
     after=$($RUFF check $files --output-format concise 2>/dev/null | grep -c ': [A-Z][0-9]')
     [ "$after" -le "$before" ] || fail "ruff: было $before, стало $after" ;;
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
