from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "wine_classification_results"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PROCESSED_DATA_DIR = RESULTS_DIR / "processed_data"

MAX_PREDICTION_TEXT_LENGTH = 1000
HITL_VERIFIED_FILENAME = "hitl_verified_labels.csv"

SECRET_KEY = "wine-classification-mvp-local-dev"
