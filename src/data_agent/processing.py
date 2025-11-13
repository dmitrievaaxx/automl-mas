# --- Утилиты DataAgent для анализа датасетов и сборки препроцессинга
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype


@dataclass
class TransformationResult:
    features: pd.DataFrame
    target: pd.Series
    pipeline: ColumnTransformer
    history: List[str]
    feature_names: List[str]


# --- Формирует краткое описание датасета для LLM
def summarize(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    columns_info = []
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        columns_info.append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_pct": round(series.isna().mean() * 100, 2),
                "unique": int(series.nunique(dropna=True)),
            }
        )

    target_distribution = (
        df[target].value_counts(normalize=True, dropna=False).round(3).to_dict()
        if target in df.columns
        else {}
    )

    denominator = df.shape[0] * df.shape[1]
    overall_missing = round((df.isna().sum().sum() / denominator) * 100, 2) if denominator else 0.0

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "columns_info": columns_info,
        "missing_pct_overall": overall_missing,
        "target_distribution": target_distribution,
        "target_dtype": str(df[target].dtype) if target in df.columns else None,
    }
    return summary


# --- Собирает промпт с инструкциями для LLM
def build_prompt(dataset_name: str, summary: Dict[str, Any], target: str) -> str:
    header = (
        "You are a senior data preprocessing expert. Produce a JSON plan for transforming the dataset.\n"
        f"Dataset: {dataset_name}\n"
        f"Rows: {summary['rows']}, Columns: {summary['columns']} (including target)\n"
        f"Target column: {target} (dtype: {summary.get('target_dtype')})\n"
        f"Target distribution: {summary.get('target_distribution')}\n"
        f"Overall missing %: {summary.get('missing_pct_overall')}%\n"
        "\n"
        "Instructions (follow ALL):\n"
        "- Respond with a SINGLE valid JSON object using keys: drop_columns, encode, normalize, impute.\n"
        "- Never include the target column in any list. Do not drop, encode, normalize or impute it.\n"
        "- drop_columns: identifiers, free-text fields, columns with >100 unique values, or >60% missing values.\n"
        "- encode: categorical columns with >2 unique categories (use one-hot). For columns with exactly two categories, map them to a single numeric column (e.g., 0/1) instead of producing multiple one-hot fields. Exclude anything already listed in drop_columns.\n"
        "- normalize: numeric columns (float/int) that benefit from scaling. Exclude target and columns slated for drop or encode.\n"
        "- impute: only columns that need missing-value handling. Allowed strategies: mean, median, most_frequent, constant.\n"
        "- Avoid conflicting actions (e.g., a column cannot be both dropped and encoded).\n"
        "- Prefer removing extremely high-cardinality text columns instead of encoding them.\n"
        "- If unsure, leave a list empty.\n"
        "\n"
        "Valid JSON example (structure only):\n"
        "{\n"
        "  \"drop_columns\": [\"<column_to_drop>\"],\n"
        "  \"encode\": [\"<categorical_column>\"],\n"
        "  \"normalize\": [\"<numeric_column>\"],\n"
        "  \"impute\": {\"<column_with_missing>\": \"median\"}\n"
        "}\n"
        "\n"
        "Columns summary (first 20):\n"
    )

    column_lines = []
    for info in summary["columns_info"][:20]:
        column_lines.append(
            f"- {info['name']}: dtype={info['dtype']}, missing%={info['missing_pct']}, unique={info['unique']}"
        )
    return header + "\n".join(column_lines)


# --- Применяет рекомендации и обучает пайплайн трансформаций
def apply_recommendations(
    df: pd.DataFrame, target: str, recommendations: Mapping[str, Any]
) -> TransformationResult:
    recommendations = _fill_defaults(recommendations)
    working = df.copy()
    history: List[str] = []

    to_drop = [col for col in recommendations["drop_columns"] if col in working.columns and col != target]
    if to_drop:
        working = working.drop(columns=to_drop)
        history.append(f"Dropped columns: {', '.join(to_drop)}")

    if target not in working.columns:
        raise KeyError(f"Target column '{target}' is missing after preprocessing.")

    X = working.drop(columns=[target])
    y = working[target]

    encode_cols = _resolve_encode_columns(X, recommendations["encode"])
    normalize_cols = [col for col in recommendations["normalize"] if col in X.columns]
    binary_maps: Dict[str, Dict[Any, int]] = {}
    for col in list(encode_cols):
        series = X[col]
        uniques = sorted(series.dropna().unique(), key=lambda value: str(value))
        if 0 < len(uniques) <= 2:
            mapping = {value: idx for idx, value in enumerate(uniques)}
            X[col] = series.map(mapping).astype(float)
            binary_maps[col] = mapping
    if binary_maps:
        encode_cols = [col for col in encode_cols if col not in binary_maps]
    impute_map = _sanitize_impute_map(
        {col: strategy for col, strategy in recommendations["impute"].items() if col in X.columns}
    )

    pipeline = _build_pipeline(X, encode_cols, normalize_cols, impute_map)
    transformed = pipeline.fit_transform(X)
    feature_names = list(pipeline.get_feature_names_out())
    features = pd.DataFrame(transformed, index=X.index, columns=feature_names)

    if binary_maps:
        for column in binary_maps:
            if column in features.columns:
                features[column] = features[column].round().astype(int)

    history.append(f"Encoded columns: {', '.join(encode_cols) if encode_cols else 'none'}")
    history.append(f"Normalized columns: {', '.join(normalize_cols) if normalize_cols else 'none'}")
    history.append(
        "Imputation strategies: "
        + (", ".join(f"{col}={strategy}" for col, strategy in impute_map.items()) if impute_map else "defaults")
    )
    if binary_maps:
        formatted = ", ".join(
            f"{col} ({' / '.join(f'{k}->{v}' for k, v in mapping.items())})" for col, mapping in binary_maps.items()
        )
        history.append(f"Binary encoded (0/1): {formatted}")

    return TransformationResult(features=features, target=y, pipeline=pipeline, history=history, feature_names=feature_names)


# --- Делит данные на train/val/test с учётом стратификации
def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, tuple[pd.DataFrame, pd.Series]]:
    stratify_target = target if _can_stratify(target, test_size, val_size) else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, target, test_size=test_size, stratify=stratify_target, random_state=random_state
    )

    adjusted_val = val_size / (1 - test_size)
    stratify_temp = y_temp if stratify_target is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val, stratify=stratify_temp, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


# --- Формирует JSON с метаданными обработки
def generate_metadata(
    dataset_name: str,
    summary: Dict[str, Any],
    result: TransformationResult,
    recommendations: Mapping[str, Any],
    prompt: str,
    llm_info: Mapping[str, Any],
    config: Mapping[str, Any],
    source_path: str,
    target: str,
    run_started: str,
    task_info: Mapping[str, Any],
    feature_roles: Mapping[str, List[str]],
    automl_exports: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    categorical_count = sum(1 for feature in result.feature_names if feature.startswith("cat_"))
    roles_payload = {key: sorted(list(value)) for key, value in feature_roles.items()}
    roles_payload["target"] = target
    exports_payload = deepcopy(automl_exports)
    task_details = dict(task_info)
    metadata = {
        "dataset": dataset_name,
        "source_path": source_path,
        "target": target,
        "rows": summary["rows"],
        "original_columns": summary["columns"],
        "processed_features": len(result.feature_names),
        "categorical_ratio": round(categorical_count / max(len(result.feature_names), 1), 3),
        "missing_pct_overall": summary["missing_pct_overall"],
        "target_distribution": summary.get("target_distribution"),
        "transformations": result.history,
        "recommendations": recommendations,
        "prompt": prompt,
        "run": {
            "started_at": run_started,
            "split": dict(config),
        },
        "task_type": task_details.get("type", "unknown"),
        "task_details": task_details,
        "auto_ml": {
            "target_column": target,
            "roles": roles_payload,
            "exports": exports_payload,
        },
        "llm": {
            "model": llm_info.get("model"),
            "status": llm_info.get("status"),
            "candidate_models": list(llm_info.get("candidate_models", [])),
            "attempts": list(llm_info.get("attempts", [])),
        },
    }
    raw_response = llm_info.get("raw_response") if isinstance(llm_info, Mapping) else None
    if raw_response:
        metadata["llm"]["raw_response"] = raw_response
    return metadata


# --- Определяет тип задачи и характеристики таргета
def infer_task_info(target: pd.Series) -> Dict[str, Any]:
    series = target.dropna()
    if series.empty:
        return {"type": "unknown", "classes": []}

    unique_values = pd.unique(series)
    unique_count = int(len(unique_values))

    if is_bool_dtype(series) or not is_numeric_dtype(series) or (
        _is_integer_like(series) and unique_count <= 20
    ):
        classes_sorted = sorted(unique_values, key=lambda value: str(value))
        classes = [_to_python_scalar(value) for value in classes_sorted]
        task_mode = "binary" if len(classes) == 2 else "multiclass"
        return {
            "type": "classification",
            "mode": task_mode,
            "classes": classes,
            "class_count": len(classes),
        }

    return {"type": "regression"}


# --- Выводит роли признаков для авто-ML
def derive_feature_roles(features: pd.DataFrame) -> Dict[str, List[str]]:
    numeric: List[str] = []
    categorical: List[str] = []
    for column, dtype in features.dtypes.items():
        if is_numeric_dtype(dtype):
            numeric.append(column)
        else:
            categorical.append(column)
    return {
        "all": sorted(list(features.columns)),
        "numeric": sorted(numeric),
        "category": sorted(categorical),
    }


# --- Преобразует значения в JSON-совместимые типы
def _to_python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


# --- Проверяет, можно ли считать числовой таргет целочисленным
def _is_integer_like(series: pd.Series) -> bool:
    if is_integer_dtype(series):
        return True
    if not is_numeric_dtype(series):
        return False
    values = series.dropna().to_numpy()
    if values.size == 0:
        return False
    return np.allclose(values, np.round(values))


# --- Заполняет пропущенные ключи рекомендаций значениями по умолчанию
def _fill_defaults(recommendations: Mapping[str, Any]) -> MutableMapping[str, Any]:
    defaults = {
        "drop_columns": [],
        "encode": [],
        "normalize": [],
        "impute": {},
        "new_features": [],
    }
    merged: MutableMapping[str, Any] = {**defaults, **recommendations}
    for key in defaults:
        if key not in merged or merged[key] is None:
            merged[key] = defaults[key]
    return merged


# --- Нормализует стратегии импутации к поддерживаемым значениям
def _sanitize_impute_map(impute_map: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    synonym_map = {
        "mode": "most_frequent",
        "mostfrequent": "most_frequent",
        "most frequent": "most_frequent",
        "most-frequent": "most_frequent",
        "mean": "mean",
        "median": "median",
        "constant": "constant",
    }
    for column, strategy in impute_map.items():
        if isinstance(strategy, str):
            lowered = strategy.strip().lower()
            normalized[column] = synonym_map.get(lowered, strategy if lowered == strategy else lowered)
        else:
            normalized[column] = strategy
    return normalized


# --- Определяет список признаков для кодирования
def _resolve_encode_columns(df: pd.DataFrame, encode: Iterable[str]) -> List[str]:
    encode_set = {col for col in encode if col in df.columns}
    inferred = {col for col in df.columns if df[col].dtype == "object" or str(df[col].dtype) == "category"}
    return sorted(encode_set | inferred)


# --- Собирает ColumnTransformer с числовыми и категориальными пайплайнами
def _build_pipeline(
    features: pd.DataFrame,
    encode_cols: Iterable[str],
    normalize_cols: Iterable[str],
    impute_map: Mapping[str, str],
) -> ColumnTransformer:
    transformers: List[tuple[str, Pipeline, List[str]]] = []
    normalize_set = set(normalize_cols)
    encode_set = set(encode_cols)

    for column in features.columns:
        column_dtype = features[column].dtype
        if column in encode_set or _is_categorical_dtype(column_dtype):
            transformers.append((f"cat_{column}", _categorical_pipeline(column, impute_map), [column]))
        else:
            transformers.append(
                (f"num_{column}", _numeric_pipeline(column, normalize_set, impute_map), [column])
            )

    column_transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return column_transformer


# --- Формирует пайплайн для категориального признака
def _categorical_pipeline(column: str, impute_map: Mapping[str, str]) -> Pipeline:
    strategy = impute_map.get(column, "most_frequent")
    imputer_kwargs = {"strategy": strategy}
    if strategy == "constant":
        imputer_kwargs["fill_value"] = "missing"
    encoder = _one_hot_encoder()
    return Pipeline([("imputer", SimpleImputer(**imputer_kwargs)), ("encoder", encoder)])


# --- Формирует пайплайн для числового признака
def _numeric_pipeline(column: str, normalize_cols: set[str], impute_map: Mapping[str, str]) -> Pipeline:
    strategy = impute_map.get(column, "median")
    imputer_kwargs = {"strategy": strategy}
    if strategy == "constant":
        imputer_kwargs["fill_value"] = 0
    steps: List[tuple[str, Any]] = [("imputer", SimpleImputer(**imputer_kwargs))]
    if column in normalize_cols:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


# --- Проверяет, является ли тип признака категориальным
def _is_categorical_dtype(dtype) -> bool:
    return str(dtype) in {"object", "category", "bool"}


# --- Создаёт OneHotEncoder с защитой от несовместимых версий
def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# --- Определяет, можно ли выполнять стратифицированное разбиение
def _can_stratify(target: pd.Series, test_size: float, val_size: float) -> bool:
    counter = Counter(target.dropna())
    class_count = len(counter)
    if class_count <= 1:
        return False
    if min(counter.values()) < 2:
        return False

    total = len(target)
    test_samples = max(1, int(round(total * test_size)))
    if test_samples < class_count:
        return False

    remaining = total - test_samples
    if remaining <= 0:
        return False
    adjusted_val = val_size / (1 - test_size) if test_size < 1 else 0
    val_samples = max(1, int(round(remaining * adjusted_val)))
    if val_samples < class_count:
        return False

    return True


__all__ = [
    "TransformationResult",
    "summarize",
    "build_prompt",
    "apply_recommendations",
    "split_dataset",
    "generate_metadata",
    "infer_task_info",
    "derive_feature_roles",
]
