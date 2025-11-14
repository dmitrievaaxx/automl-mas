from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .models import AutomlContext, Recommendation


def write_report(
    dataset_name: str,
    context: AutomlContext,
    recommendations: List[Recommendation],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{dataset_name}_research.json"

    payload = {
        "dataset": dataset_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "metric": context.metric_name,
            "value": context.metric_value,
            "best_model": {
                "name": context.best_model_name,
                "score": context.best_model_score,
            },
        },
        "recommendations": [
            {
                "title": rec.title,
                "description": rec.description,
                "source": rec.source,
                "url": rec.url,
                "expected_gain": rec.expected_gain,
                "complexity": rec.complexity,
            }
            for rec in recommendations
        ],
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return report_path

