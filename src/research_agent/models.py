from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AutomlContext:
    dataset: str
    metric_name: str
    metric_value: float
    task_type: str
    best_model_name: str
    best_model_score: Optional[float]
    processed_features: List[str] = field(default_factory=list)


@dataclass
class SearchQuery:
    topic: str
    text: str


@dataclass
class SearchResult:
    source: str
    title: str
    url: str
    summary: str
    score: Optional[float] = None


@dataclass
class Recommendation:
    title: str
    description: str
    source: str
    url: str
    expected_gain: Optional[str] = None
    complexity: Optional[str] = None
    score: Optional[float] = None

