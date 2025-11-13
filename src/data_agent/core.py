# --- Управляет жизненным циклом DataAgent и его конфигурацией
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Mapping, Optional

from copy import deepcopy

from .io_utils import load_multiple, reset_processed_folder, save_automl_formats, save_metadata
from .llm_client import OpenRouterLLM
from .processing import (
    apply_recommendations,
    build_prompt,
    derive_feature_roles,
    generate_metadata,
    infer_task_info,
    split_dataset,
    summarize,
)


@dataclass
class AgentConfig:
    val_size: float = 0.2
    test_size: float = 0.1
    random_state: int = 42


class DataAgent:
    # --- Инициализирует агента набором датасетов, таргетом и LLM-клиентом
    def __init__(
        self,
        datasets: Mapping[str, str],
        target: str,
        llm_client: Optional[OpenRouterLLM] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        if not datasets:
            raise ValueError("Provide at least one dataset.")
        self.datasets = datasets
        self.target = target
        self.llm = llm_client
        self.config = config or AgentConfig()

    # --- Запускает обработку датасетов и возвращает пути артефактов
    def run(self) -> Dict[str, Dict[str, str]]:
        data_frames = load_multiple(self.datasets)
        results: Dict[str, Dict[str, str]] = {}

        for name, df in data_frames.items():
            reset_processed_folder(name)
            run_started = datetime.utcnow().isoformat() + "Z"
            summary = summarize(df, self.target)
            prompt = build_prompt(name, summary, self.target)
            recommendations = self._get_recommendations(name, prompt)
            llm_info = {}
            if self.llm is not None and getattr(self.llm, "last_call", None):
                llm_info = deepcopy(self.llm.last_call)
            transformation = apply_recommendations(df, self.target, recommendations)

            splits = split_dataset(
                transformation.features,
                transformation.target,
                val_size=self.config.val_size,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            automl_exports = save_automl_formats(name, splits, self.target)

            split_config = {
                "val_size": self.config.val_size,
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
            }
            task_info = infer_task_info(transformation.target)
            feature_roles = derive_feature_roles(transformation.features)
            metadata = generate_metadata(
                dataset_name=name,
                summary=summary,
                result=transformation,
                recommendations=recommendations,
                prompt=prompt,
                llm_info=llm_info,
                config=split_config,
                source_path=str(self.datasets[name]),
                target=self.target,
                run_started=run_started,
                task_info=task_info,
                feature_roles=feature_roles,
                automl_exports=automl_exports,
            )
            metadata_path = save_metadata(name, metadata)

            results[name] = {
                "dataset_name": name,
                "metadata_path": metadata_path,
                "task_type": task_info.get("type"),
                "automl": automl_exports,
            }

        return results

    # --- Запрашивает рекомендации по препроцессингу у LLM
    def _get_recommendations(self, dataset_name: str, prompt: str) -> Dict:
        if self.llm is None:
            raise RuntimeError("LLM client is not configured. Provide OpenRouterLLM instance.")
        return self.llm.get_recommendations(dataset_name, prompt)

