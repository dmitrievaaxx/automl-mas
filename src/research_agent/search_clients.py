from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import requests
from xml.etree import ElementTree as ET

from .models import SearchQuery, SearchResult


DEFAULT_TIMEOUT = 15


@dataclass
class BaseClient:
    name: str

    def search(self, query: SearchQuery, limit: int = 5) -> List[SearchResult]:
        raise NotImplementedError


class TavilyClient(BaseClient):
    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(name="Tavily")
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def search(self, query: SearchQuery, limit: int = 5) -> List[SearchResult]:
        if not self.api_key:
            return []

        url = "https://api.tavily.com/search"
        payload = {
            "query": query.text,
            "search_depth": "advanced",
            "max_results": limit,
            "api_key": self.api_key,
        }

        try:
            response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException:
            return []

        data = response.json()
        results: List[SearchResult] = []
        for item in data.get("results", [])[:limit]:
            results.append(
                SearchResult(
                    source=self.name,
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    summary=item.get("content", item.get("snippet", "")),
                )
            )
        return results


class ArxivClient(BaseClient):
    def __init__(self) -> None:
        super().__init__(name="arXiv")

    def search(self, query: SearchQuery, limit: int = 5) -> List[SearchResult]:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query.text}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        }

        try:
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException:
            return []

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results: List[SearchResult] = []
        for entry in root.findall("atom:entry", ns)[:limit]:
            title = entry.findtext("atom:title", default="", namespaces=ns)
            summary = entry.findtext("atom:summary", default="", namespaces=ns)
            link_elem = entry.find("atom:id", ns)
            link = link_elem.text if link_elem is not None else ""

            results.append(
                SearchResult(
                    source=self.name,
                    title=title.strip(),
                    url=link.strip(),
                    summary=summary.strip(),
                )
            )

        return results


class GitHubClient(BaseClient):
    def __init__(self, token: Optional[str] = None) -> None:
        super().__init__(name="GitHub")
        self.token = token or os.getenv("GITHUB_TOKEN")

    def search(self, query: SearchQuery, limit: int = 5) -> List[SearchResult]:
        if not self.token:
            return []

        url = "https://api.github.com/search/repositories"
        params = {
            "q": query.text,
            "sort": "stars",
            "order": "desc",
            "per_page": limit,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ResearchAgent",
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException:
            return []

        payload = response.json()
        results: List[SearchResult] = []
        for item in payload.get("items", [])[:limit]:
            results.append(
                SearchResult(
                    source=self.name,
                    title=item.get("full_name", ""),
                    url=item.get("html_url", ""),
                    summary=item.get("description") or "",
                    score=float(item.get("stargazers_count") or 0),
                )
            )
        return results


def get_default_clients() -> List[BaseClient]:
    clients: List[BaseClient] = [TavilyClient(), ArxivClient(), GitHubClient()]
    return clients

