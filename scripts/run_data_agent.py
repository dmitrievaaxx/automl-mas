# --- Запуск DataAgent с автообнаружением датасетов
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from data_agent.core import DataAgent
from data_agent.llm_client import OpenRouterLLM


# --- Приводит путь к датасету к абсолютному виду
def resolve_dataset_path(raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path.resolve()
    candidate = (PROJECT_ROOT / raw_path).resolve()
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / "data" / "raw" / raw_path).resolve()


# --- Собирает список файлов-датасетов из аргументов или каталога data/raw
def discover_dataset_files(dataset_args: list[str] | None) -> dict[str, Path]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    datasets: dict[str, Path] = {}

    if dataset_args:
        for item in dataset_args:
            resolved = resolve_dataset_path(Path(item))
            datasets[_dataset_key(resolved, raw_root)] = resolved
        return datasets

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data folder not found at {raw_root}")

    for dataset_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        base_name = dataset_dir.name
        candidate = (dataset_dir / "train.csv").resolve()
        if candidate.exists():
            datasets[f"{base_name}_train"] = candidate

    if not datasets:
        raise FileNotFoundError("No train CSV files discovered in data/raw/<dataset>/ directories.")

    return datasets


# --- Определяет ключ набора данных на основе файла и каталога raw
def _dataset_key(csv_path: Path, raw_root: Path) -> str:
    csv_path = csv_path.resolve()
    try:
        relative = csv_path.parent.relative_to(raw_root)
    except ValueError:
        return csv_path.stem

    parts = relative.parts
    if not parts or parts == (".",):
        return csv_path.stem

    base = parts[-1]
    if csv_path.stem == "train":
        return f"{base}_{csv_path.stem}"
    return csv_path.stem


# --- Основная точка входа агента
def main() -> None:
    dataset_map = discover_dataset_files(None)
    try:
        llm = OpenRouterLLM()
    except ValueError as exc:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Set the environment variable before running the agent."
        ) from exc

    agent = DataAgent(
        datasets=dataset_map,
        target="Survived",
        llm_client=llm,
    )

    result = agent.run()

    for dataset_name in result:
        print(
            f"{dataset_name} обработан: CSV в data/processed/{dataset_name}, metadata в reports."
        )


if __name__ == "__main__":
    main()

