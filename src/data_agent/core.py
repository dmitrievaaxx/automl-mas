# --- Управляет жизненным циклом DataAgent и его конфигурацией
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Mapping, Optional

from copy import deepcopy

import pandas as pd

from .io_utils import (
    load_metadata,
    load_multiple,
    reset_processed_folder,
    save_automl_formats,
    save_metadata,
)
from .llm_client import OpenRouterLLM
from .processing import (
    apply_recommendations,
    build_prompt,
    derive_feature_roles,
    generate_metadata,
    infer_task_info,
    summarize,
)


@dataclass
class AgentConfig:
    val_size: float = 0.0
    test_size: float = 0.0
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

        grouped: Dict[str, Dict[str, tuple[str, pd.DataFrame]]] = {}
        for name, df in data_frames.items():
            base_name, role = self._parse_dataset_name(name)
            if role == "train" and self.target not in df.columns:
                role = "test"
            grouped.setdefault(base_name, {})[role] = (name, df)

        for base_name, split_map in grouped.items():
            if "train" not in split_map:
                raise ValueError(f"Для набора '{base_name}' не найден обучающий файл (train).")

            train_label, train_df = split_map["train"]
            reset_processed_folder(base_name)

            run_started = datetime.utcnow().isoformat() + "Z"
            summary = summarize(train_df, self.target)
            prompt = build_prompt(base_name, summary, self.target)
            recommendations = self._get_recommendations(base_name, prompt)
            llm_info = {}
            if self.llm is not None and getattr(self.llm, "last_call", None):
                llm_info = deepcopy(self.llm.last_call)

            transformation = apply_recommendations(train_df, self.target, recommendations)

            combined = transformation.features.copy()
            combined[self.target] = transformation.target.values
            automl_exports = save_automl_formats(
                base_name,
                combined,
                self.target,
                split_name="train",
                reset=True,
            )

            split_config = {
                "val_size": self.config.val_size,
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
            }
            task_info = infer_task_info(transformation.target)
            feature_roles = derive_feature_roles(transformation.features)
            metadata = generate_metadata(
                dataset_name=base_name,
                summary=summary,
                result=transformation,
                recommendations=recommendations,
                prompt=prompt,
                llm_info=llm_info,
                config=split_config,
                source_path=str(self.datasets[train_label]),
                target=self.target,
                run_started=run_started,
                task_info=task_info,
                feature_roles=feature_roles,
                automl_exports=automl_exports,
            )
            metadata_path = save_metadata(base_name, metadata)

            results[train_label] = {
                "dataset_name": base_name,
                "metadata_path": metadata_path,
                "task_type": task_info.get("type"),
                "automl": automl_exports,
                "split": "train",
            }

        return results

    # --- Запрашивает рекомендации по препроцессингу у LLM
    def _get_recommendations(self, dataset_name: str, prompt: str) -> Dict:
        if self.llm is None:
            raise RuntimeError("LLM client is not configured. Provide OpenRouterLLM instance.")
        return self.llm.get_recommendations(dataset_name, prompt)

    @staticmethod
    def _parse_dataset_name(dataset_name: str) -> tuple[str, str]:
        if dataset_name.endswith("_train"):
            return dataset_name[:-6], "train"
        return dataset_name, "train"

