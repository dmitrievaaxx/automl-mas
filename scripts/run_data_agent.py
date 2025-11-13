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
            candidates = _collect_csv_candidates(resolved)
            if not candidates:
                raise FileNotFoundError(f"No CSV files found for dataset input: {item}")
            for csv_path in candidates:
                datasets[_derive_dataset_name(csv_path, raw_root)] = csv_path
        return datasets

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data folder not found at {raw_root}")

    for entry in sorted(raw_root.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".csv":
            csv_path = entry.resolve()
            datasets[_derive_dataset_name(csv_path, raw_root)] = csv_path
        elif entry.is_dir():
            candidates = _collect_csv_candidates(entry)
            if len(candidates) == 1:
                csv_path = candidates[0]
                datasets[_derive_dataset_name(csv_path, raw_root)] = csv_path
            elif len(candidates) > 1:
                raise RuntimeError(
                    f"Multiple CSV files found in {entry}. Specify --dataset explicitly for this folder."
                )

    if not datasets:
        raise FileNotFoundError(f"No CSV datasets discovered under {raw_root}")

    return datasets


# --- Подбирает подходящие CSV внутри переданного пути
def _collect_csv_candidates(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path.resolve()]
    if path.is_dir():
        specific = path / f"{path.name}.csv"
        if specific.exists():
            return [specific.resolve()]
        train = path / "train.csv"
        if train.exists():
            return [train.resolve()]
        csv_files = sorted(p.resolve() for p in path.glob("*.csv"))
        return csv_files
    return []


# --- Определяет имя набора данных по файлу и каталогу raw
def _derive_dataset_name(csv_path: Path, raw_root: Path) -> str:
    try:
        relative_parent = csv_path.parent.resolve().relative_to(raw_root.resolve())
        if relative_parent == Path("."):
            return csv_path.stem
        return relative_parent.parts[-1]
    except ValueError:
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
        print(f"{dataset_name} обработан: сохранены splits и metadata в data/processed/{dataset_name}.")


if __name__ == "__main__":
    main()

