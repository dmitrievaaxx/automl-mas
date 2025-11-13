# --- Утилиты ввода-вывода и работы с путями для DataAgent
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# --- Создаёт директорию, если её ещё нет
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Очищает папку и создаёт её заново
def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return ensure_dir(path)


# --- Загружает датасет из файла или URL в DataFrame
def load_dataset(source: str | Path) -> pd.DataFrame:
    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".json"}:
            return pd.read_json(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported file format for {source}")

    source_str = str(source)
    if source_str.startswith("http"):
        return pd.read_csv(source_str)

    raise FileNotFoundError(f"Dataset source not found: {source}")


# --- Загружает несколько наборов данных в словарь
def load_multiple(datasets: Mapping[str, str | Path]) -> Dict[str, pd.DataFrame]:
    return {name: load_dataset(path) for name, path in datasets.items()}


# --- Сохраняет датасеты в форматах, подходящих для LAMA и FEDOT
def save_automl_formats(
    dataset_name: str,
    splits: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    target_column: str,
) -> Dict[str, Dict[str, Any]]:
    dataset_dir = reset_dir(PROCESSED_DIR / dataset_name)
    lama_dir = ensure_dir(dataset_dir / "LAMA")
    fedot_dir = ensure_dir(dataset_dir / "FEDOT")

    lama_info: Dict[str, Any] = {
        "format": "csv",
        "target_column": target_column,
        "base_dir": str(lama_dir.resolve()),
        "splits": {},
    }
    fedot_info: Dict[str, Any] = {
        "format": "csv",
        "target_column": target_column,
        "data_type": "table",
        "base_dir": str(fedot_dir.resolve()),
        "splits": {},
    }

    for split_name, (features, target) in splits.items():
        target_name = target.name or target_column

        lama_path = lama_dir / f"{dataset_name}_{split_name}.csv"
        combined = features.copy()
        if target_name in combined.columns:
            combined = combined.drop(columns=[target_name])
        combined[target_name] = target.values
        combined.to_csv(lama_path, index=False)
        lama_info["splits"][split_name] = str(lama_path)

        x_path = fedot_dir / f"{dataset_name}_X_{split_name}.csv"
        y_path = fedot_dir / f"{dataset_name}_y_{split_name}.csv"
        features.to_csv(x_path, index=False)
        target.to_csv(y_path, index=False, header=True)
        fedot_info["splits"][f"X_{split_name}"] = str(x_path)
        fedot_info["splits"][f"y_{split_name}"] = str(y_path)

    return {"lama": lama_info, "fedot": fedot_info}


# --- Сохраняет метаданные обработки в JSON-файл
def save_metadata(dataset_name: str, metadata: Dict) -> str:
    base_dir = ensure_dir(PROCESSED_DIR / dataset_name)
    meta_path = base_dir / f"{dataset_name}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return str(meta_path)


# --- Полностью очищает папку обработанных данных
def reset_processed_folder(dataset_name: str) -> Path:
    base_dir = PROCESSED_DIR / dataset_name
    if base_dir.exists():
        shutil.rmtree(base_dir)
    return ensure_dir(base_dir)

