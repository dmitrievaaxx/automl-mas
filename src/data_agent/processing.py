# Набор инструментов для анализа датасета и построения пайплайна препроцессинга
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TransformationResult:
    features: pd.DataFrame
    target: pd.Series
    pipeline: ColumnTransformer
    history: List[str]
    feature_names: List[str]


# Формирует краткое описание датасета для LLM
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


# Собирает промпт с инструкциями для LLM
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
        "- encode: categorical columns with <=50 unique values requiring one-hot encoding. Exclude anything listed in drop_columns.\n"
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


# Применяет рекомендации и обучает пайплайн трансформаций
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
    impute_map = _sanitize_impute_map(
        {col: strategy for col, strategy in recommendations["impute"].items() if col in X.columns}
    )

    pipeline = _build_pipeline(X, encode_cols, normalize_cols, impute_map)
    transformed = pipeline.fit_transform(X)
    feature_names = list(pipeline.get_feature_names_out())
    features = pd.DataFrame(transformed, index=X.index, columns=feature_names)

    history.append(f"Encoded columns: {', '.join(encode_cols) if encode_cols else 'none'}")
    history.append(f"Normalized columns: {', '.join(normalize_cols) if normalize_cols else 'none'}")
    history.append(
        "Imputation strategies: "
        + (", ".join(f"{col}={strategy}" for col, strategy in impute_map.items()) if impute_map else "defaults")
    )

    return TransformationResult(features=features, target=y, pipeline=pipeline, history=history, feature_names=feature_names)


# Делит данные на train/val/test с учётом стратификации
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


# Формирует JSON с метаданными обработки
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
) -> Dict[str, Any]:
    categorical_count = sum(1 for feature in result.feature_names if feature.startswith("cat_"))
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


# Заполняет пропущенные ключи рекомендаций значениями по умолчанию
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


# Нормализует стратегии импутации к поддерживаемым значениям
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


# Создаёт новые признаки по выражениям, если это возможно
def _create_new_features(df: pd.DataFrame, new_features: Sequence[Any]) -> List[str]:
    created: List[str] = []
    for feature in new_features:
        if isinstance(feature, dict) and {"name", "expression"} <= feature.keys():
            name = feature["name"]
            expression = feature["expression"]
        elif isinstance(feature, str) and "=" in feature:
            name, expression = [part.strip() for part in feature.split("=", 1)]
        else:
            continue

        if not name:
            continue
        try:
            df[name] = df.eval(expression)
            created.append(name)
        except Exception:
            continue
    return created


# Определяет список признаков для кодирования
def _resolve_encode_columns(df: pd.DataFrame, encode: Iterable[str]) -> List[str]:
    encode_set = {col for col in encode if col in df.columns}
    inferred = {col for col in df.columns if df[col].dtype == "object" or str(df[col].dtype) == "category"}
    return sorted(encode_set | inferred)


# Собирает ColumnTransformer с числовыми и категориальными пайплайнами
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


# Формирует пайплайн для категориального признака
def _categorical_pipeline(column: str, impute_map: Mapping[str, str]) -> Pipeline:
    strategy = impute_map.get(column, "most_frequent")
    imputer_kwargs = {"strategy": strategy}
    if strategy == "constant":
        imputer_kwargs["fill_value"] = "missing"
    encoder = _one_hot_encoder()
    return Pipeline([("imputer", SimpleImputer(**imputer_kwargs)), ("encoder", encoder)])


# Формирует пайплайн для числового признака
def _numeric_pipeline(column: str, normalize_cols: set[str], impute_map: Mapping[str, str]) -> Pipeline:
    strategy = impute_map.get(column, "median")
    imputer_kwargs = {"strategy": strategy}
    if strategy == "constant":
        imputer_kwargs["fill_value"] = 0
    steps: List[tuple[str, Any]] = [("imputer", SimpleImputer(**imputer_kwargs))]
    if column in normalize_cols:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


# Проверяет, является ли тип признака категориальным
def _is_categorical_dtype(dtype) -> bool:
    return str(dtype) in {"object", "category", "bool"}


# Создаёт OneHotEncoder с защитой от несовместимых версий
def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# Определяет, можно ли выполнять стратифицированное разбиение
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


# Безопасно сериализует JSON дедуплицируя Unicode
def _json_dumps_safe(payload: Any) -> str:
    import json
    return json.dumps(payload, indent=2, ensure_ascii=False)
