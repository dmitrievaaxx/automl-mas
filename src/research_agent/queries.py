from __future__ import annotations

from typing import List

from .models import AutomlContext, SearchQuery


TEMPLATES = [
    "best model for {dataset} tabular {task} roc auc",
    "{model} improvements for {dataset} dataset",
    "boosting alternatives to {model} for tabular data",
    "{dataset} dataset feature engineering roc auc improvement",
    "state of the art AutoML tabular {task} roc auc {year}",
]


def build_queries(context: AutomlContext, year: int) -> List[SearchQuery]:
    topic_map = {
        0: "general performance",
        1: "model specific",
        2: "alternative models",
        3: "feature engineering",
        4: "latest research",
    }
    queries: List[SearchQuery] = []
    for idx, template in enumerate(TEMPLATES):
        text = template.format(
            dataset=context.dataset,
            task=context.task_type,
            model=context.best_model_name,
            year=year,
        )
        queries.append(SearchQuery(topic=topic_map[idx], text=text))
    return queries

