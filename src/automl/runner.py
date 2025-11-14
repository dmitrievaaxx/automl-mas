#!/usr/bin/env python
"""
AutoML Runner для обучения LightAutoML и формирования JSON-отчёта.

Модуль читает метаданные Data-Agent, загружает подготовленные CSV из splits/,
запускает LightAutoML и сохраняет детальный отчёт с метриками и информацией о лучшей модели.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_LAMA = PROJECT_ROOT / "external_libs" / "lightautoml"
if EXTERNAL_LAMA.exists() and str(EXTERNAL_LAMA) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_LAMA))

try:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
except ImportError as exc:
    raise ImportError(
        "Не удалось импортировать LightAutoML. Проверьте, установлен ли пакет и PyTorch."
    ) from exc

try:
    from lightautoml.utils import create_leaderboard  # отсутствует в старых версиях
except ImportError:
    create_leaderboard = None


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
VariantName = Literal["baseline", "research"]


def load_metadata(dataset_name: str) -> Dict[str, Any]:
    """Читает метадату Data-Agent для датасета."""
    metadata_path = REPORTS_DIR / f"{dataset_name}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Метадата не найдена: {metadata_path}")
    
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_splits(dataset_name: str, splits_info: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Загружает train/val/test CSV из splits/."""
    files = splits_info.get("files", {})
    result = {}
    
    for split_name in ["train", "val", "test"]:
        if split_name not in files:
            warnings.warn(f"Отсутствует {split_name} split для {dataset_name}")
            continue
        
        path = Path(files[split_name])
        if not path.exists():
            warnings.warn(f"Файл не найден: {path}")
            continue
        
        result[split_name] = pd.read_csv(path)
    
    return result


def _resolve_preset_class(variant: VariantName) -> Type[TabularAutoML]:
    if variant == "research":
        try:
            from lightautoml.automl.presets.tabular_presets import ResearchTabularAutoML

            return ResearchTabularAutoML
        except ImportError:
            warnings.warn("ResearchTabularAutoML не найден, используется базовый TabularAutoML.")
    return TabularAutoML


def train_automl(
    train_df: pd.DataFrame,
    target_column: str,
    task_mode: str,
    timeout: Optional[int] = 600,
    verbose: bool = True,
    variant: VariantName = "baseline",
) -> tuple[TabularAutoML, Any]:
    """
    Обучает TabularAutoML и возвращает модель + OOF-прогнозы.
    
    Args:
        train_df: обучающий датасет
        target_column: имя таргета
        task_type: тип задачи (classification/regression)
        timeout: таймаут обучения в секундах
    
    Returns:
        automl: обученный пресет
        oof_predictions: out-of-fold прогнозы для формирования leaderboard
    """
    if target_column not in train_df.columns:
        raise KeyError(f"Столбец таргета '{target_column}' не найден в данных")
    task = Task(task_mode)

    preset_cls = _resolve_preset_class(variant)
    reader_params = {"n_jobs": 1, "advanced_roles": False}
    selection_params = {"mode": 0}
    tuning_params = {"max_tuning_iter": 0, "max_tuning_time": 0}
    automl = preset_cls(
        task=task,
        timeout=timeout,
        reader_params=reader_params,
        selection_params=selection_params,
        tuning_params=tuning_params,
    )

    if verbose:
        print(f"▶️  Запуск LightAutoML ({task_mode}, timeout={timeout}s)...")
    oof_predictions = automl.fit_predict(train_df, roles={"target": target_column})
    
    return automl, oof_predictions


def evaluate_model(
    automl: TabularAutoML,
    test_df: pd.DataFrame,
    target_column: str,
) -> Dict[str, float]:
    """Оценивает модель на тестовой выборке, используя встроенную метрику LightAutoML."""
    if target_column not in test_df.columns:
        raise KeyError(f"Столбец таргета '{target_column}' отсутствует в тестовых данных.")

    missing_mask = test_df[target_column].isna()
    if missing_mask.any():
        removed = int(missing_mask.sum())
        warnings.warn(
            f"Удалено {removed} строк с пропущенным значением таргета '{target_column}' перед расчётом метрик."
        )
        test_df = test_df.loc[~missing_mask].reset_index(drop=True)
        if test_df.empty:
            warnings.warn("После удаления пропусков в таргете не осталось данных для оценки.")
            return {}

    metric_func = getattr(automl.task, "metric_func", None)
    metric_name = getattr(automl.task, "metric_name", None)

    if metric_func is None:
        warnings.warn("LightAutoML Task не предоставляет metric_func — метрики недоступны.")
        return {}

    features = test_df.drop(columns=[target_column])
    target = test_df[target_column].values
    predictions = automl.predict(features).data
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]

    metrics: Dict[str, float] = {}
    try:
        score = metric_func(target, predictions)
        metric_label = str(metric_name or getattr(getattr(metric_func, "func", metric_func), "__name__", "metric"))
        metrics[metric_label] = float(score)
    except Exception as exc:
        warnings.warn(f"Не удалось вычислить метрику LightAutoML '{metric_name}': {exc}")

    return metrics


def create_report(
    dataset_name: str,
    metadata: Dict[str, Any],
    automl: TabularAutoML,
    oof_predictions: Any,
    test_metrics: Dict[str, float],
    timeout: Optional[int],
    variant: VariantName,
) -> Dict[str, Any]:
    """Формирует JSON-отчёт с результатами AutoML."""
    
    # Пытаемся получить leaderboard
    leaderboard_data = None
    best_model_info = {}
    
    try:
        if create_leaderboard:
            leaderboard = create_leaderboard(oof_predictions)
            if leaderboard is not None and not leaderboard.empty:
                leaderboard_data = leaderboard.to_dict(orient="records")
                best_model = leaderboard.iloc[0]
                best_model_info = {
                    "name": str(best_model.get("model", "unknown")),
                    "score": float(best_model.get("score", 0.0)) if best_model.get("score") is not None else None,
                }
    except Exception as e:
        warnings.warn(f"Не удалось создать leaderboard через lightautoml.utils: {e}")
    
    if leaderboard_data is None:
        fallback_entries = _build_fallback_leaderboard(automl)
        if fallback_entries:
            leaderboard_data = fallback_entries
            best_model_info = {
                "name": fallback_entries[0]["model"],
                "weight": fallback_entries[0]["weight"],
            }
    
    if not best_model_info:
        primary_metric = _primary_metric(metadata.get("task_type", "classification"))
        best_model_info = {
            "name": "LightAutoML Tabular Ensemble",
            "metric": primary_metric,
            "score": test_metrics.get(primary_metric),
        }

    if best_model_info.get("score") is None:
        metric_key = None
        metric_value = None
        if test_metrics:
            metric_key, metric_value = next(iter(test_metrics.items()))
        if metric_value is not None:
            best_model_info["score"] = metric_value
            if "metric" not in best_model_info:
                best_model_info["metric"] = metric_key

    report = {
        "dataset": dataset_name,
        "task_type": metadata.get("task_type", "unknown"),
        "target_column": metadata.get("target", "unknown"),
        "automl": {
            "data": {
                "train": metadata.get("auto_ml", {}).get("exports", {}).get("splits", {}).get("files", {}).get("train"),
                "test": metadata.get("auto_ml", {}).get("exports", {}).get("splits", {}).get("files", {}).get("test"),
            },
            "framework": "LightAutoML",
            "version": "0.4.1",  # можно динамически получать
            "variant": variant,
            "run": {
                "started_at": datetime.utcnow().isoformat() + "Z",
                "timeout_seconds": timeout,
                "variant": variant,
            },
            "best_model": best_model_info,
            "test_metrics": test_metrics,
            "leaderboard": leaderboard_data,
        },
        "data_processing": {
            "rows": metadata.get("rows"),
            "original_columns": metadata.get("original_columns"),
            "processed_features": metadata.get("processed_features"),
            "transformations": metadata.get("transformations", []),
        },
    }
    
    return report


def save_report(dataset_name: str, report: Dict[str, Any]) -> str:
    """Сохраняет отчёт в JSON-файл."""
    output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_automl.json"
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return str(output_path)


def run(
    dataset_name: str,
    timeout: Optional[int] = 600,
    verbose: bool = True,
    variant: VariantName = "baseline",
) -> Dict[str, Any]:
    """
    Главная функция: запускает полный пайплайн AutoML для датасета.
    
    Args:
        dataset_name: имя датасета (например, 'titanic')
        timeout: таймаут обучения в секундах
    
    Returns:
        словарь с результатами (путь к отчёту, метрики и т.д.)
    """
    if verbose:
        print(f"📊 Загрузка метаданных для {dataset_name}...")
    metadata = load_metadata(dataset_name)
    
    target_column = metadata.get("target")
    task_type = metadata.get("task_type", "classification")
    splits_info = metadata.get("auto_ml", {}).get("exports", {}).get("splits", {})
    
    if not target_column:
        raise ValueError("В метаданных отсутствует информация о таргете")
    
    if verbose:
        print(f"📂 Загрузка splits (train/val/test)...")
        splits = load_splits(dataset_name, splits_info)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            splits = load_splits(dataset_name, splits_info)
    
    if "train" not in splits:
        raise ValueError("Отсутствует train split — перезапустите Data-Agent.")

    train_df = splits["train"]
    test_df = splits.get("test")

    if test_df is None:
        warnings.warn("Тестовый сплит не найден. Сформирован holdout 20% из train.csv.")
        train_df, test_df = _make_holdout(train_df, target_column)
    
    # Обучение
    task_mode = _map_task_type(task_type, metadata.get("task_details"))

    automl, oof_predictions = train_automl(
        train_df, target_column, task_mode, timeout, verbose=verbose, variant=variant
    )
    
    # Оценка на тесте
    if verbose:
        print(f"📈 Оценка на тестовой выборке...")
    if verbose:
        test_metrics = evaluate_model(automl, test_df, target_column)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            test_metrics = evaluate_model(automl, test_df, target_column)
    
    # Формирование отчёта
    report = create_report(
        dataset_name, metadata, automl, oof_predictions, test_metrics, timeout, variant
    )
    
    # Сохранение
    report_path = save_report(dataset_name, report)
    
    if verbose:
        print(f"✅ Готово! Отчёт сохранён: {report_path}")
        print(f"🏆 Лучшая модель: {report['automl']['best_model'].get('name', 'N/A')}")
        print(f"📊 Тестовые метрики: {test_metrics}")
    
    return {
        "dataset": dataset_name,
        "report_path": report_path,
        "test_metrics": test_metrics,
        "best_model": report["automl"]["best_model"],
        "variant": variant,
    }


def _map_task_type(task_type: str, task_details: Optional[Dict[str, Any]]) -> str:
    """Преобразует информацию о типе задачи в формат LAMA (binary/multiclass/reg)."""
    if task_details:
        mode = str(task_details.get("mode", "")).lower()
        class_count = task_details.get("class_count")
        if mode == "binary" or class_count == 2:
            return "binary"
        if mode == "multiclass" or (isinstance(class_count, int) and class_count and class_count > 2):
            return "multiclass"
        if str(task_details.get("type", "")).lower().startswith("reg"):
            return "reg"

    task_lower = task_type.lower()
    if "binary" in task_lower:
        return "binary"
    if "multiclass" in task_lower:
        return "multiclass"
    if "reg" in task_lower:
        return "reg"
    # по умолчанию бинарная классификация
    return "binary"


def _primary_metric(task_type: str) -> str:
    """Возвращает ключевую метрику для указанного типа задачи."""
    return "rmse" if "reg" in task_type.lower() else "roc_auc"


def _build_fallback_leaderboard(automl: TabularAutoML) -> Optional[list[Dict[str, Any]]]:
    """Строит fallback-лидерборд из текстового описания ансамбля."""
    try:
        desc = automl.create_model_str_desc()
    except Exception as exc:
        warnings.warn(f"Не удалось получить описание модели LightAutoML: {exc}")
        return None

    entries: list[Dict[str, Any]] = []
    for line in desc.splitlines():
        line = line.strip()
        if not line or line.startswith("Final prediction"):
            continue
        if line.endswith("+"):
            line = line[:-1].rstrip()
        if "*" not in line:
            continue
        weight_part, model_part = line.split("*", 1)
        try:
            weight = float(weight_part.strip())
        except ValueError:
            continue

        model_part = model_part.strip()
        if model_part.startswith("(") and model_part.endswith(")"):
            model_part = model_part[1:-1]
        entries.append(
            {
                "model": model_part,
                "weight": weight,
            }
        )

    # отсортируем по весу убыванию
    entries.sort(key=lambda x: x["weight"], reverse=True)
    return entries or None


def _make_holdout(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Создаёт holdout-разбиение, сохраняя стратификацию для классификации."""
    if target_column not in df.columns:
        raise KeyError(f"Столбец таргета '{target_column}' отсутствует в данных.")

    target = df[target_column]
    stratify = None
    value_counts = target.value_counts(dropna=False)
    if value_counts.size > 1 and value_counts.min() >= 2:
        stratify = target

    train_part, test_part = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )
    return train_part.reset_index(drop=True), test_part.reset_index(drop=True)

