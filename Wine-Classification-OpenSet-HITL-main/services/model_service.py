import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from config import MAX_PREDICTION_TEXT_LENGTH, MODELS_DIR
from services.data_service import get_experiment_config, get_osr_config, safe_load_pkl


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:MAX_PREDICTION_TEXT_LENGTH]


def _load_first_pickle(names: list[str]) -> Any | None:
    for name in names:
        model = safe_load_pkl(MODELS_DIR / name)
        if model is not None:
            return model
    return None


@lru_cache(maxsize=1)
def load_models() -> dict[str, Any]:
    classifier = _load_first_pickle(["classifier.pkl", "wine_classifier.pkl"])
    label_encoder = _load_first_pickle(["label_encoder.pkl"])
    osr_config = get_osr_config()
    return {
        "classifier": classifier,
        "label_encoder": label_encoder,
        "osr_config": osr_config,
        "available": classifier is not None and label_encoder is not None,
    }


@lru_cache(maxsize=1)
def get_bert_model() -> Any | None:
    config = get_experiment_config()
    model_name = (
        config.get("bert_model")
        or config.get("sentence_transformer_model")
        or config.get("embedding_model")
        or "all-MiniLM-L6-v2"
    )
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception:
        return None


def generate_embedding(text: str) -> np.ndarray | None:
    model = get_bert_model()
    if model is None:
        return None
    try:
        embedding = model.encode([clean_text(text)], convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(embedding)
    except Exception:
        return None


def _threshold(osr_config: dict[str, Any]) -> float:
    for key in ("threshold", "osr_threshold", "confidence_threshold", "optimal_threshold"):
        value = osr_config.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.5


def predict_wine(text: str) -> dict[str, Any]:
    cleaned = clean_text(text)
    if not cleaned:
        return {"error": "Please enter a wine name or description."}

    models = load_models()
    if not models["available"]:
        return {"error": "Live model prediction is unavailable. Precomputed experiment results are still available below."}

    embedding = generate_embedding(cleaned)
    if embedding is None:
        return {"error": "Live model prediction is unavailable. The SentenceTransformer embedding model could not be loaded."}

    classifier = models["classifier"]
    label_encoder = models["label_encoder"]
    threshold = _threshold(models["osr_config"])

    try:
        probabilities = classifier.predict_proba(embedding)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        closest_class = label_encoder.inverse_transform([best_index])[0]
        closest_class = str(closest_class)
    except Exception as exc:
        return {"error": f"Live model prediction failed: {exc}"}

    if confidence < threshold:
        return {
            "predicted_class": "UNKNOWN",
            "closest_known_class": closest_class,
            "confidence": confidence,
            "osr_status": "UNKNOWN",
            "classification_status": "Rejected by OSR",
            "threshold": threshold,
        }

    return {
        "predicted_class": closest_class,
        "confidence": confidence,
        "osr_status": "KNOWN",
        "classification_status": "Accepted known wine key",
        "threshold": threshold,
    }


def live_prediction_status() -> dict[str, Any]:
    models = load_models()
    return {
        "available": bool(models["available"]),
        "message": "" if models["available"] else "Live model prediction is unavailable. Precomputed experiment results are still available below.",
    }
