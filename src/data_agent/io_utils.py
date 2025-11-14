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
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


# --- Сохраняет датасеты в формате splits (train/val/test)
def save_automl_formats(
    dataset_name: str,
    dataset: pd.DataFrame,
    target_column: str,
    split_name: str = "train",
    reset: bool = False,
) -> Dict[str, Dict[str, Any]]:
    dataset_dir = PROCESSED_DIR / dataset_name
    if reset:
        reset_dir(dataset_dir)
    else:
        ensure_dir(dataset_dir)

    split_path = dataset_dir / f"{split_name}.csv"
    dataset.to_csv(split_path, index=False)

    splits_info: Dict[str, Any] = {
        "format": "csv",
        "target_column": target_column,
        "base_dir": str(dataset_dir.resolve()),
        "files": {
            split_name: str(split_path)
        },
    }

    return {"splits": splits_info}


# --- Сохраняет метаданные обработки в JSON-файл
def save_metadata(dataset_name: str, metadata: Dict) -> str:
    reports_dir = ensure_dir(PROJECT_ROOT / "reports")
    meta_path = reports_dir / f"{dataset_name}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return str(meta_path)


# --- Загружает готовые метаданные
def load_metadata(dataset_name: str) -> Dict[str, Any]:
    meta_path = PROJECT_ROOT / "reports" / f"{dataset_name}_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}")
    with meta_path.open("r", encoding="utf-8") as file:
        return json.load(file)


# --- Полностью очищает папку обработанных данных
def reset_processed_folder(dataset_name: str) -> Path:
    base_dir = PROCESSED_DIR / dataset_name
    if base_dir.exists():
        shutil.rmtree(base_dir)
    return ensure_dir(base_dir)

