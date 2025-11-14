from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .models import AutomlContext, Recommendation, SearchResult


def _guess_complexity(source: str) -> str:
    mapping = {
        "Tavily": "research summary",
        "GitHub": "implementation available",
        "arXiv": "theory-heavy",
    }
    return mapping.get(source, "unknown")


def aggregate_recommendations(
    context: AutomlContext,
    results: Iterable[SearchResult],
) -> List[Recommendation]:
    grouped: Dict[str, List[SearchResult]] = defaultdict(list)
    for result in results:
        grouped[result.source].append(result)

    recommendations: List[Recommendation] = []
    for source, bucket in grouped.items():
        for item in bucket:
            expected_gain = None
            if context.metric_value:
                expected_gain = f"aim to beat {context.metric_name}={context.metric_value:.4f}"
            recommendations.append(
                Recommendation(
                    title=item.title,
                    description=item.summary,
                    source=source,
                    url=item.url,
                    expected_gain=expected_gain,
                    complexity=_guess_complexity(source),
                    score=item.score,
                )
            )
    recommendations.sort(key=lambda rec: rec.score or 0.0, reverse=True)
    return recommendations

