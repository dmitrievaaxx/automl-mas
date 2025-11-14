#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from research_agent.runner import run  # noqa: E402


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, sep, value = stripped.partition("=")
            if sep and key and value:
                os.environ.setdefault(key.strip(), value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск ResearchAgent для поиска улучшений.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Название датасета (используется для отчёта из reports/<dataset>_automl.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Количество результатов на запрос для Tavily.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    if not os.getenv("TAVILY_API_KEY"):
        print("TAVILY_API_KEY не задан. Укажи ключ в .env или переменной окружения.")
        sys.exit(1)
    if not os.getenv("GITHUB_TOKEN"):
        print("⚠️  GITHUB_TOKEN не задан. GitHub-поиск будет пропущен.")

    report_path = run(dataset_name=args.dataset, per_query_limit=args.limit)
    print(f"Research report saved to {report_path}")


if __name__ == "__main__":
    main()

