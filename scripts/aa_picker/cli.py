"""CLI entrypoint for Artificial Analysis model picker."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .fetch_and_store import ensure_fresh_db
from .picker import format_telegram_text, pick_models


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aa_picker",
        description="Подбор ИИ-модели по базе Artificial Analysis + Gemini Decision.",
    )
    p.add_argument("task", nargs="*", help="Описание задачи свободным текстом")
    p.add_argument(
        "--priority",
        choices=["auto", "quality", "price", "speed"],
        default="auto",
        help="Приоритет подбора (по умолчанию auto через Gemini Decision)",
    )
    p.add_argument("--max-price", type=float, default=None, help="Макс. цена input $/1M токенов")
    p.add_argument("--min-speed", type=float, default=None, help="Мин. скорость вывода (tok/s)")
    p.add_argument("--max-ttft", type=float, default=None, help="Макс. время до первого токена (сек)")
    p.add_argument("--min-context", type=int, default=None, help="Мин. размер контекста (токенов)")
    p.add_argument("--all", action="store_true", dest="show_all", help="Показать все модели, не приоритизировать только парк")
    p.add_argument("--refresh", action="store_true", help="Принудительно обновить базу с artificialanalysis.ai")
    p.add_argument("--json", action="store_true", dest="as_json", help="Вывести полный JSON")
    p.add_argument("--db", default=None, help="Путь к SQLite-базе")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    task_text = " ".join(args.task).strip()

    if args.refresh and not task_text:
        info = ensure_fresh_db(db_path=args.db, force=True)
        if args.as_json:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            print(
                f"База обновлена: {info.get('models_count')} моделей, "
                f"{info.get('endpoints_count', 0)} эндпоинтов, "
                f"путь: {info.get('db_path')}"
            )
        return 0

    if not task_text:
        parser.error("Укажите описание задачи или --refresh для обновления базы.")

    result = pick_models(
        task_text,
        priority=args.priority,
        max_price=args.max_price,
        min_speed=args.min_speed,
        max_ttft=args.max_ttft,
        min_context=args.min_context,
        show_all=args.show_all,
        refresh=args.refresh,
        db_path=args.db,
    )

    if args.as_json:
        result["telegram_text"] = format_telegram_text(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_telegram_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
