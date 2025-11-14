#!/usr/bin/env python
import argparse
import os
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

warnings.filterwarnings("ignore", category=UserWarning)

from automl.runner import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск AutoML (baseline/research).")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Название датасета из data/processed/<dataset>.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Таймаут LightAutoML в секундах (None = без ограничения).",
    )
    parser.add_argument(
        "--variant",
        choices=("baseline", "research", "both"),
        default="baseline",
        help="Какой режим запустить: стандартный, исследовательский или оба последовательно.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Пробрасывать verbose=True в automl.runner.run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["LIGHTAUTO_QUIET"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning)

    variants = ["baseline", "research"] if args.variant == "both" else [args.variant]
    results = []

    for variant in variants:
        print(f"▶️  AutoML variant: {variant}")
        result = run(
            dataset_name=args.dataset,
            timeout=args.timeout,
            verbose=args.verbose,
            variant=variant,
        )
        results.append((variant, result))
        print(f"   ↳ best model: {result['best_model'].get('name', 'N/A')}")
        print(f"   ↳ metrics   : {result['test_metrics']}")

    if len(results) == 2:
        base_metrics = results[0][1]["test_metrics"]
        research_metrics = results[1][1]["test_metrics"]
        print("📊 Сравнение baseline vs research:")
        for metric in set(base_metrics) | set(research_metrics):
            base_val = base_metrics.get(metric)
            research_val = research_metrics.get(metric)
            if base_val is None or research_val is None:
                print(f"  - {metric}: baseline={base_val}, research={research_val}")
                continue
            delta = research_val - base_val
            print(f"  - {metric}: baseline={base_val:.5f}, research={research_val:.5f}, Δ={delta:+.5f}")


if __name__ == "__main__":
    main()

