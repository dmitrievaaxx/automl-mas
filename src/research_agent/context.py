from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AutomlContext


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def _extract_features(metadata: Dict[str, Any]) -> List[str]:
    section = metadata.get("data_processing", {})
    features = section.get("processed_features")
    if isinstance(features, list):
        return [str(f) for f in features]
    return []


def _read_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_context(dataset_name: str) -> AutomlContext:
    report_path = REPORTS_DIR / f"{dataset_name}_automl.json"
    if not report_path.exists():
        raise FileNotFoundError(f"АвтоML отчёт не найден: {report_path}")

    with report_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    automl_section: Dict[str, Any] = payload.get("automl", {})
    best_model: Dict[str, Any] = automl_section.get("best_model", {})

    metric_key, metric_value = None, None
    test_metrics = automl_section.get("test_metrics", {})
    if isinstance(test_metrics, dict) and test_metrics:
        metric_key, metric_value = next(iter(test_metrics.items()))

    context = AutomlContext(
        dataset=payload.get("dataset", dataset_name),
        metric_name=str(metric_key or "metric"),
        metric_value=_read_float(metric_value) or 0.0,
        task_type=payload.get("task_type", "classification"),
        best_model_name=str(best_model.get("name", "unknown")),
        best_model_score=_read_float(best_model.get("score")),
        processed_features=_extract_features(payload),
    )
    return context

