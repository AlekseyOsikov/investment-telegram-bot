"""Проверка: каждый успешный эталон превращается в команду, которую понимает бот.

Использует настоящий `price_watch.parse.parse_watch_args` из соседнего проекта
`investment-telegram-bot`. Если проект недоступен, проверка пропускается.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BOT_SRC = Path(__file__).resolve().parents[3] / "src"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DATA / "v1" / "train.jsonl"))
    parser.add_argument("--valid", default=str(DATA / "v1" / "valid.jsonl"))
    parser.add_argument("--eval", default=str(DATA / "eval.jsonl"))
    args = parser.parse_args()

    if not (BOT_SRC / "price_watch" / "parse.py").exists():
        print(f"пропуск: не найден {BOT_SRC / 'price_watch' / 'parse.py'}")
        return
    sys.path.insert(0, str(BOT_SRC))
    from price_watch.parse import WatchRequest, parse_watch_args  # type: ignore

    checked = 0
    for path in (Path(args.train), Path(args.valid), Path(args.eval)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            expected = row["meta"]["expected"]
            if not expected.get("ok"):
                continue
            command = "/watch " + " ".join(expected["secids"]) + f" {expected['poll_interval']} {expected['report_interval']}"
            parsed = parse_watch_args(command.split()[1:])
            assert isinstance(parsed, WatchRequest), f"{path.name}: {command} → {parsed}"
            assert set(parsed.secids) == set(expected["secids"]), (command, parsed.secids, expected["secids"])
            assert parsed.poll_interval == expected["poll_interval"], (command, parsed.poll_interval)
            assert parsed.report_interval == expected["report_interval"], (command, parsed.report_interval)
            checked += 1

    print(f"совместимо с ботом: {checked} успешных эталонов превращаются в валидную команду /watch")


if __name__ == "__main__":
    main()
