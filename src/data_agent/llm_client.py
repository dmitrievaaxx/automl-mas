# Клиент OpenRouter с поддержкой фолбэков и логированием вызовов
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests import Response


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_MODELS = [
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "deepseek/deepseek-r1-0528-qwen3-8b:free",
    "qwen/qwen3-coder:free",
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]


class OpenRouterLLM:
    # Настраивает клиента, список моделей и HTTP-сессию
    def __init__(
        self,
        api_key: Optional[str] = None,
        models: Optional[Iterable[str]] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 60,
    ) -> None:
        if load_dotenv is not None:
            load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Set the OpenRouter API key via constructor or OPENROUTER_API_KEY environment variable.")
        self.models: List[str] = list(models) if models else list(DEFAULT_MODELS)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.last_call: Optional[Dict[str, Any]] = None

    # Перебирает модели, пока одна не вернёт корректный JSON
    def get_recommendations(self, dataset_name: str, prompt: str) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        self.last_call = {
            "dataset": dataset_name,
            "prompt": prompt,
            "candidate_models": list(self.models),
            "attempts": [],
            "status": "pending",
        }
        for model in self.models:
            try:
                raw = self._call_model(model, prompt)
                self.last_call["attempts"].append({"model": model, "status": "success"})
                self.last_call.update({"model": model, "raw_response": raw, "status": "success"})
                return self._parse_json(raw)
            except Exception as exc:
                self.last_call["attempts"].append({"model": model, "status": "error", "error": str(exc)})
                last_error = exc
                continue
        self.last_call["status"] = "failed"
        self.last_call["error"] = str(last_error) if last_error else "Unknown error"
        raise RuntimeError(f"All LLM models failed: {last_error}") from last_error

    # Выполняет HTTP-вызов выбранной модели OpenRouter
    def _call_model(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a data preparation assistant. Reply with JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = self.session.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
        )
        self._raise_for_status(response)
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices returned by LLM response.")
        return choices[0]["message"]["content"]

    # Преобразует строку ответа в словарь JSON
    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        candidates = [raw.strip()]
        if "```" in raw:
            stripped = raw.strip("`")
            for delimiter in ("json", "JSON"):
                stripped = stripped.removeprefix(f"```{delimiter}").removesuffix("```").strip()
            candidates.append(stripped)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise ValueError("LLM response is not valid JSON.")

    # Бросает подробную ошибку, если HTTP-ответ неуспешный
    @staticmethod
    def _raise_for_status(response: Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"OpenRouter request failed: {detail}") from exc

