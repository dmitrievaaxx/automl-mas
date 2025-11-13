#!/usr/bin/env python
"""
Скрипт для запуска AutoML-пайплайна на подготовленных данных.

Использует модуль automl.runner для обучения LightAutoML и формирования отчёта.
Настройте параметры датасета и таймаута прямо в коде.
"""

import os
import sys
import warnings
from pathlib import Path

# Добавляем src в path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

warnings.filterwarnings("ignore", category=UserWarning)

from automl.runner import run


def main() -> None:
    """Главная точка входа."""
    
    # Параметры (настраивайте здесь)
    DATASET_NAME = "titanic"
    TIMEOUT = 600  # секунды, None = без ограничения
    
    # Включаем подавление UserWarning внутри LightAutoML
    os.environ["LIGHTAUTO_QUIET"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        result = run(dataset_name=DATASET_NAME, timeout=TIMEOUT, verbose=False)
        print(f"Лучшая модель: {result['best_model'].get('name', 'N/A')}")
        print(f"Тестовые метрики: {result['test_metrics']}")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()

