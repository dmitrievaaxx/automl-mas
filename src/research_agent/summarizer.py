from __future__ import annotations

import json
from typing import List

from data_agent.llm_client import OpenRouterLLM

from .models import AutomlContext, Recommendation, SearchResult


PROMPT_TEMPLATE = """
You are an AutoML research analyst. A dataset named "{dataset}" currently achieves {metric_name} = {metric_value:.4f}.
Below is a JSON array of candidate sources discussing ways to improve this dataset. 
Your task:
1. Summarize each candidate (if multiple sources are identical, merge them).
2. Propose only methods that are likely to beat the baseline metric. 
3. For each keep: 
   - "title": short name of the method or paper,
   - "url": source link,
   - "source": type (arXiv, Tavily, GitHub, etc.),
   - "idea": concise summary (max 2 sentences),
   - "expected_metric": estimated ROC AUC (float) if applied to this dataset (or null if impossible),
   - "expected_gain": textual delta like "+0.02 AUC" (optional),
   - "difficulty": easy / medium / hard,
   - "score": number 0..1 reflecting potential impact,
   - "recommended": true/false (true only if expected_metric > baseline or you are confident it will beat baseline).
4. Discard sources explicitly reporting metrics below baseline.

Return strict JSON:
{{"recommendations": [ ... ]}}

Candidate sources:
{sources}
"""


def summarize_with_llm(
    context: AutomlContext,
    results: List[SearchResult],
    max_items: int = 10,
) -> List[Recommendation]:
    if not results:
        return []

    items = []
    for result in results[:max_items]:
        items.append(
            {
                "title": result.title,
                "summary": result.summary,
                "url": result.url,
                "source": result.source,
            }
        )

    prompt = PROMPT_TEMPLATE.format(
        dataset=context.dataset,
        metric_name=context.metric_name,
        metric_value=context.metric_value,
        sources=json.dumps(items, ensure_ascii=False),
    )

    llm = OpenRouterLLM()
    response = llm.get_recommendations(context.dataset, prompt)
    entries = response.get("recommendations", [])
    recommendations: List[Recommendation] = []

    for entry in entries:
        if not entry:
            continue
        if entry.get("recommended") is False:
            continue

        expected_metric = _parse_float(entry.get("expected_metric"))
        if expected_metric is not None and expected_metric <= context.metric_value:
            continue

        recommendations.append(
            Recommendation(
                title=entry.get("title") or entry.get("method") or "Unnamed approach",
                description=entry.get("idea") or entry.get("summary", ""),
                source=entry.get("source", "LLM"),
                url=entry.get("url", ""),
                expected_gain=entry.get("expected_gain"),
                complexity=entry.get("difficulty"),
                score=_parse_float(entry.get("score")),
            )
        )

    recommendations.sort(key=lambda rec: rec.score or 0.0, reverse=True)
    return recommendations


def _parse_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None



