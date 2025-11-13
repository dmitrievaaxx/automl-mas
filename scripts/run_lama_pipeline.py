#!/usr/bin/env python
"""
Запускает LightAutoML на подготовленных CSV с признаками для задачи классификации.

По умолчанию использует датасет Titanic, подготовленный DataAgent и сохранённый в
`data/processed/<dataset_name>/LAMA`. Скрипт запускает табличный пресет LAMA,
обучает модель на train+val и оценивает качество на тестовой выборке с метрикой ROC-AUC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from lightautoml.utils import create_leaderboard
except ImportError:
    create_leaderboard = None

try:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
except ImportError as exc:
    raise SystemExit(
        "LightAutoML не установлен. Активируйте виртуальное окружение (или убедитесь, что пакет установлен) "
        "и повторите запуск: `pip install lightautoml`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "titanic"
TARGET_COLUMN = "Survived"


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает train/val/test CSV для указанного датасета."""
    dataset_dir = PROJECT_ROOT / "data" / "processed" / dataset_name / "LAMA"
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Каталог {dataset_dir} не найден. Сначала сгенерируйте данные через DataAgent."
        )

    train_path = dataset_dir / f"{dataset_name}_train.csv"
    val_path = dataset_dir / f"{dataset_name}_val.csv"
    test_path = dataset_dir / f"{dataset_name}_test.csv"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Ожидался файл {path}, но он отсутствует.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    timeout: int | None = 600,
) -> None:
    """Обучает TabularAutoML и печатает ROC-AUC на тестовой выборке."""
    train_full = pd.concat([train_df, val_df], ignore_index=True)

    if target_column not in train_full.columns:
        raise KeyError(f"Столбец таргета `{target_column}` не найден в данных.")

    task = Task("binary")
    automl = TabularAutoML(task=task, timeout=timeout)

    print("▶️  Запускаем обучение LightAutoML...")
    oof_predictions = automl.fit_predict(train_full, roles={"target": target_column})

    leaderboard = create_leaderboard(oof_predictions) if create_leaderboard else None
    if leaderboard is not None and not leaderboard.empty:
        best_model = leaderboard.iloc[0]
        try:
            score_value = float(best_model.get("score", float("nan")))
            print(
                f"🏆 Лучший модельный бленд: {best_model.get('model', 'unknown')} "
                f"(score={score_value:.4f})."
            )
        except (TypeError, ValueError):
            print(f"🏆 Лучший модельный бленд: {best_model.get('model', 'unknown')}.")
        print("Топ-5 моделей по версии LightAutoML:")
        print(leaderboard.head())

    features_test = test_df.drop(columns=[target_column])
    target_test = test_df[target_column]

    predictions = automl.predict(features_test).data[:, 0]
    score = roc_auc_score(target_test, predictions)
    print(f"✅ Готово. ROC-AUC на тестовой выборке: {score:.4f}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск LightAutoML на подготовленных табличных данных."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Имя датасета в `data/processed` (по умолчанию {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Таймаут обучения в секундах. 0 или отрицательное значение выключает ограничение.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    timeout = args.timeout if args.timeout and args.timeout > 0 else None

    train_df, val_df, test_df = load_dataset(args.dataset)
    train_and_evaluate(train_df, val_df, test_df, timeout=timeout)


if __name__ == "__main__":
    main()

