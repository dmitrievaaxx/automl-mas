from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .aggregator import aggregate_recommendations
from .context import REPORTS_DIR, load_context
from .models import Recommendation, SearchResult
from .queries import build_queries
from .report import write_report
from .search_clients import BaseClient, get_default_clients
from .summarizer import summarize_with_llm


def _collect_results(
    queries,
    clients: Iterable[BaseClient],
    per_query_limit: int,
) -> List[SearchResult]:
    results: List[SearchResult] = []
    for query in queries:
        for client in clients:
            entries = client.search(query, limit=per_query_limit)
            results.extend(entries)
    return results


def run(
    dataset_name: str = "titanic",
    per_query_limit: int = 5,
    output_dir: Path | None = None,
    clients: Iterable[BaseClient] | None = None,
) -> Path:
    context = load_context(dataset_name)
    queries = build_queries(context, year=2025)

    active_clients = list(clients) if clients else get_default_clients()
    search_results = _collect_results(queries, active_clients, per_query_limit)

    recommendations: List[Recommendation] = []
    try:
        recommendations = summarize_with_llm(context, search_results)
    except Exception:
        recommendations = aggregate_recommendations(context, search_results)

    target_dir = output_dir or REPORTS_DIR
    report_path = write_report(dataset_name, context, recommendations, target_dir)
    return report_path

