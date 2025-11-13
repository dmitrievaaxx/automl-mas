from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Mapping, Tuple

import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# Создаёт директорию при необходимости и возвращает путь
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# Загружает датасет из локального файла или URL в DataFrame
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


# Загружает несколько датасетов в словарь DataFrame
def load_multiple(datasets: Mapping[str, str | Path]) -> Dict[str, pd.DataFrame]:
    return {name: load_dataset(path) for name, path in datasets.items()}


# Сохраняет разбиения train/val/test в CSV и возвращает пути
def save_splits(dataset_name: str, splits: Mapping[str, Tuple[pd.DataFrame, pd.Series]]) -> Dict[str, str]:
    base_dir = ensure_dir(PROCESSED_DIR / dataset_name)
    paths: Dict[str, str] = {}
    for split_name, (features, target) in splits.items():
        x_path = base_dir / f"{dataset_name}_X_{split_name}.csv"
        y_path = base_dir / f"{dataset_name}_y_{split_name}.csv"
        features.to_csv(x_path, index=False)
        target.to_csv(y_path, index=False, header=True)
        paths[f"X_{split_name}"] = str(x_path)
        paths[f"y_{split_name}"] = str(y_path)
    return paths


# Сохраняет метаданные в JSON-файл
def save_metadata(dataset_name: str, metadata: Dict) -> str:
    base_dir = ensure_dir(PROCESSED_DIR / dataset_name)
    meta_path = base_dir / f"{dataset_name}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return str(meta_path)


# Очищает папку с обработанными данными для повторного запуска
def reset_processed_folder(dataset_name: str) -> Path:
    base_dir = PROCESSED_DIR / dataset_name
    if base_dir.exists():
        shutil.rmtree(base_dir)
    return ensure_dir(base_dir)


# Скачивает датасет соревнования Kaggle через CLI и распаковывает архив
def download_kaggle_competition(competition: str, destination: Path | None = None, force: bool = False) -> Path:
    destination = ensure_dir(destination or RAW_DIR / competition)
    archive_path = destination / f"{competition}.zip"

    if archive_path.exists() and not force:
        return destination

    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(destination)]
    subprocess.run(cmd, check=True)

    if archive_path.exists():
        shutil.unpack_archive(str(archive_path), extract_dir=destination)
    else:
        raise FileNotFoundError(f"Expected Kaggle archive at {archive_path}, but it was not created.")

    return destination

