# --- Управляет жизненным циклом DataAgent и его конфигурацией
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Mapping, Optional

from copy import deepcopy

from .io_utils import (
    load_multiple,
    reset_processed_folder,
    save_metadata,
    save_splits,
)
from .llm_client import OpenRouterLLM
from .processing import (
    apply_recommendations,
    build_prompt,
    generate_metadata,
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
            split_paths = save_splits(name, splits)

            split_config = {
                "val_size": self.config.val_size,
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
            }
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
            )
            metadata_path = save_metadata(name, metadata)

            artifact_paths = {
                **split_paths,
                "metadata_json": metadata_path,
            }

            results[name] = {
                "dataset_name": name,
                "train_path": split_paths["X_train"],
                "train_y_path": split_paths["y_train"],
                "val_path": split_paths["X_val"],
                "val_y_path": split_paths["y_val"],
                "test_path": split_paths["X_test"],
                "test_y_path": split_paths["y_test"],
                "metadata_path": metadata_path,
            }

        return results

    # --- Запрашивает рекомендации по препроцессингу у LLM
    def _get_recommendations(self, dataset_name: str, prompt: str) -> Dict:
        if self.llm is None:
            raise RuntimeError("LLM client is not configured. Provide OpenRouterLLM instance.")
        return self.llm.get_recommendations(dataset_name, prompt)

