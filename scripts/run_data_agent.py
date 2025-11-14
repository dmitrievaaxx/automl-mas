# --- Запуск DataAgent с автообнаружением датасетов
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

import argparse

from data_agent.core import DataAgent
from data_agent.llm_client import OpenRouterLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск DataAgent для одного датасета.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Название подпапки в data/raw, например titanic или German_Credit_Data.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Имя таргет-колонки в train.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = (PROJECT_ROOT / "data" / "raw" / args.dataset).resolve()
    train_csv = dataset_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv не найден по пути {train_csv}")

    try:
        llm = OpenRouterLLM()
    except ValueError as exc:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Set the environment variable before running the agent."
        ) from exc

    agent = DataAgent(
        datasets={f"{args.dataset}_train": train_csv},
        target=args.target,
        llm_client=llm,
    )

    result = agent.run()

    for dataset_name in result:
        print(
            f"{dataset_name} обработан: CSV в data/processed/{dataset_name}, metadata в reports."
        )


if __name__ == "__main__":
    main()

