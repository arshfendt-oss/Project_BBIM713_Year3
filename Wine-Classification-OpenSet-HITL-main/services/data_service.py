import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, RESULTS_DIR


MISSING_MESSAGE = "Result not available for this experiment."


def safe_read_csv(path: Path | str) -> pd.DataFrame | None:
    path = Path(path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def safe_read_json(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return None


def safe_image(filename: str | list[str]) -> str | None:
    names = filename if isinstance(filename, list) else [filename]
    for name in names:
        path = FIGURES_DIR / name
        if path.exists() and path.is_file():
            return name
    return None


def safe_load_pkl(path: Path | str) -> Any | None:
    path = Path(path)
    try:
        if not path.exists():
            return None
        return joblib.load(path)
    except Exception:
        return None


def read_first_csv(paths: list[Path]) -> pd.DataFrame | None:
    for path in paths:
        data = safe_read_csv(path)
        if data is not None:
            return data
    return None


def get_final_summary() -> dict[str, Any]:
    return safe_read_json(REPORTS_DIR / "final_summary.json") or {}


def get_experiment_config() -> dict[str, Any]:
    return safe_read_json(RESULTS_DIR / "experiment_config.json") or {}


def get_classification_metrics() -> pd.DataFrame | None:
    return safe_read_csv(REPORTS_DIR / "classification_metrics.csv")


def get_osr_metrics() -> pd.DataFrame | None:
    return safe_read_csv(REPORTS_DIR / "osr_metrics.csv")


def get_osr_config() -> dict[str, Any]:
    json_config = safe_read_json(REPORTS_DIR / "osr_config.json")
    if json_config:
        return json_config
    pkl_config = safe_load_pkl(MODELS_DIR / "osr_config.pkl")
    return pkl_config if isinstance(pkl_config, dict) else {}


def get_clustering_metrics() -> pd.DataFrame | None:
    return safe_read_csv(REPORTS_DIR / "clustering_metrics.csv")


def get_retraining_results() -> pd.DataFrame | None:
    return safe_read_csv(REPORTS_DIR / "retraining_comparison.csv")


def get_known_predictions() -> pd.DataFrame | None:
    return safe_read_csv(PROCESSED_DATA_DIR / "known_test_predictions.csv")


def get_unknown_predictions() -> pd.DataFrame | None:
    return read_first_csv([
        PROCESSED_DATA_DIR / "unknown_predictions.csv",
        PROCESSED_DATA_DIR / "external_unknown_predictions.csv",
    ])


def get_clustered_records() -> pd.DataFrame | None:
    return safe_read_csv(PROCESSED_DATA_DIR / "clustered_unknown_records.csv")


def get_hitl_records() -> pd.DataFrame | None:
    return read_first_csv([
        REPORTS_DIR / "hitl_labels.csv",
        REPORTS_DIR / "HITL_review_required.csv",
    ])


def dataframe_records(df: pd.DataFrame | None, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None:
        return []
    clean = df.fillna("")
    if limit is not None:
        clean = clean.head(limit)
    return clean.to_dict(orient="records")


def metric_map(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    columns = {str(col).lower(): col for col in df.columns}
    metric_col = columns.get("metric") or columns.get("name")
    value_col = columns.get("score") or columns.get("value")
    if metric_col and value_col:
        return {str(row[metric_col]): row[value_col] for _, row in df.iterrows()}
    if len(df.columns) >= 2:
        return {str(row[df.columns[0]]): row[df.columns[1]] for _, row in df.iterrows()}
    return {}


def format_value(value: Any, percent: bool = False) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        numeric = float(value)
        if percent:
            return f"{numeric * 100:.2f}%"
        if 0 <= numeric <= 1:
            return f"{numeric:.4f}"
        if numeric.is_integer():
            return f"{int(numeric):,}"
        return f"{numeric:,.4f}"
    except (TypeError, ValueError):
        return str(value)


def find_metric(source: dict[str, Any], candidates: list[str]) -> Any:
    normalized = {str(k).lower().replace(" ", "_").replace("-", "_"): v for k, v in source.items()}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_").replace("-", "_")
        if key in normalized:
            return normalized[key]
    return None


def available_figures() -> list[dict[str, str]]:
    descriptions = {
        "class_distribution.png": "Distribution of records across wine keys.",
        "top_wine_keys.png": "Most frequent wine keys in the dataset.",
        "text_length_distribution.png": "Length profile of wine text records.",
        "classification_metrics.png": "Known-class classifier performance.",
        "confusion_matrix.png": "Known-class prediction confusion matrix.",
        "osr_analysis.png": "Open-set recognition analysis.",
        "osr_confidence_distribution.png": "Confidence distribution for OSR decisions.",
        "osr_roc_curve.png": "ROC curve for unknown detection.",
        "cluster_visualization.png": "Semantic cluster visualization.",
        "semantic_clusters.png": "Semantic clustering projection.",
        "retraining_comparison.png": "Before and after retraining comparison.",
        "retraining_performance_comparison.png": "Experimental retraining performance comparison.",
    }
    figures = []
    if not FIGURES_DIR.exists():
        return figures
    for path in sorted(FIGURES_DIR.iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            title = path.stem.replace("_", " ").title()
            figures.append({
                "filename": path.name,
                "title": title,
                "description": descriptions.get(path.name, "Experiment figure generated by the ML pipeline."),
            })
    return figures
