"""Одноразовый раннер длинных диалогов `/smart_agent` (не часть бота).

Прогоняет сценарии по 10–15 реплик через изолированного SmartAgent с временной памятью (как
research/rag_compare.py: пустой профиль, без MCP-серверов) и после каждого ответа делает тот же
технический шаг автомата, что и командный слой. По каждому ходу проверяет (правила — в
research/dialog_eval.py): источники либо явная строка; цель задачи; сохранность собранных ключей;
откат этапа с причиной; отсутствие отказа «не знаю» без вызова модели на ходу активной задачи.

Запуск из корня проекта (нужны .env бота, провайдер основного потока и запущенный Ollama):
    python scripts/dialog_eval.py                      # все data/dialog_eval/*.json
    python scripts/dialog_eval.py data/dialog_eval/portfolio.json [...]
    python scripts/dialog_eval.py --out-dir /путь/к/отчётам

Формат сценария (файлы НЕ коммитятся: реплики привязаны к корпусу оператора, а корпус в
репозиторий не входит; каталог data/ в .gitignore):
    {"name": "Портфель на пенсию",
     "task": {"type": "portfolio", "goal": "портфель на пенсию"},
     "turns": ["реплика 1", ..., "реплика 10"]}          # от 10 до 15 реплик

Задача стартует явно (как /smart_agent_task_start), чтобы цель не зависела от автодетектора.
Отчёт (с текстами реплик и ответов!) пишется в --out-dir (по умолчанию data/dialog_eval/reports/).
Код выхода 1 — нарушена хотя бы одна обязательная проверка или прогон прерван ошибкой.
"""

import argparse
import glob
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_SCENARIOS_GLOB = os.path.join(ROOT, "data", "dialog_eval", "*.json")
DEFAULT_OUT_DIR = os.path.join(ROOT, "data", "dialog_eval", "reports")
PROFILE_NAME = "dialog_eval"
CHAT_ID = "dialog_eval"


def run_scenario(scenario, timeout: float | None):
    """Один прогон на свежем агенте; возвращает ScenarioReport."""
    # Импорты здесь: до разбора аргументов и проверки формата сценариев не нужны ни .env,
    # ни клиент провайдера (ошибка формата не должна требовать настроенного окружения).
    from agents import rag_context
    from agents.smart_agent import SmartAgent
    from providers.main_client import main_client
    from research import dialog_eval as ev

    report = ev.ScenarioReport(scenario=scenario)
    # Без автоповторов SDK: зависший провайдер не должен растягивать прогон (как rag_compare).
    client = main_client.with_options(max_retries=0)
    kwargs = {"timeout": timeout} if timeout else {}
    with tempfile.TemporaryDirectory(prefix="dialog_eval_") as tmp:
        agent = SmartAgent(
            CHAT_ID,
            client=client,
            memory_dir=tmp,
            mcp_moex_dir="",
            mcp_bybit_dir="",
            cbr_enabled=False,
            **kwargs,
        )
        agent.create_profile(PROFILE_NAME, {})
        if not agent.start_task(scenario.task_type, scenario.goal):
            report.error = "не удалось начать задачу"
            return report

        for number, question in enumerate(scenario.turns, start=1):
            before = agent.get_working()
            status, _ = agent.get_rag_status()
            rag_active = status not in (rag_context.STATUS_OFF, rag_context.STATUS_NO_INDEX)
            try:
                answer = agent.ask(question)
                agent.update_task_state()
            except Exception as error:  # noqa: BLE001 — любой сбой прерывает прогон сценария
                message = f"{type(error).__name__}: {error}"
                result = ev.check_turn(
                    ev.TurnObservation(
                        number=number, question=question, answer="", task_before=before,
                        task_after=agent.get_working(), rag_active=rag_active, error=message,
                    ),
                    scenario.goal,
                )
                report.turns.append(result)
                report.error = f"ход {number}: {message}"
                print(ev.format_turn_line(result), flush=True)
                break
            observation = ev.TurnObservation(
                number=number,
                question=question,
                answer=answer.text,
                task_before=before,
                task_after=agent.get_working(),
                rag_active=rag_active,
                sources=len(answer.rag_sources),
                citations=len(answer.citations),
                citations_unverified=answer.citations_unverified,
                no_materials_note=answer.no_materials_note,
                stage_skip_note=answer.stage_skip_note,
                abstained=answer.abstained,
                search_failed=rag_context.FAILURE_WARNING in answer.warnings,
            )
            result = ev.check_turn(observation, scenario.goal)
            report.turns.append(result)
            print(ev.format_turn_line(result), flush=True)
    return report


def save_report(report, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"{stamp}.json")
    with open(path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "scenarios", nargs="*", help="файлы сценариев (по умолчанию data/dialog_eval/*.json)"
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="каталог отчётов (вне git)")
    parser.add_argument("--timeout", type=float, default=None, help="таймаут обращения к модели, с")
    args = parser.parse_args()

    from research import dialog_eval as ev

    paths = args.scenarios or sorted(glob.glob(DEFAULT_SCENARIOS_GLOB))
    if not paths:
        print(f"Нет сценариев: положите файлы в {os.path.dirname(DEFAULT_SCENARIOS_GLOB)} "
              "или передайте пути аргументами.")
        return 2
    try:
        scenarios = [ev.load_scenario(path) for path in paths]  # весь формат — до моделей
    except ev.ScenarioError as error:
        print(f"Ошибка формата: {error}")
        return 2

    failed = False
    for scenario in scenarios:
        print(f"\n=== {scenario.name} ({scenario.task_type}, реплик {len(scenario.turns)}) ===")
        report = run_scenario(scenario, args.timeout)
        print(ev.format_summary(report))
        print(f"Отчёт: {save_report(report, args.out_dir)}")
        failed = failed or report.failed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
