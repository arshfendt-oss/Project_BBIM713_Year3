# Unknown Rejection and Clustering for Post-Training Labelling

Professional Flask MVP for a wine classification capstone project. The app reads machine-learning outputs from `wine_classification_results` and presents known-class classification, open-set recognition, semantic clustering, HITL verification, and experimental retraining results.

## Architecture

- `app.py`: Flask routes, downloads, error handlers, and figure serving.
- `services/data_service.py`: safe CSV, JSON, image, and PKL loading.
- `services/model_service.py`: BERT embedding generation and live classifier prediction.
- `services/hitl_service.py`: review queue loading and verified label export.
- `templates/`: Jinja2 pages.
- `static/css/style.css`: responsive academic dashboard styling.
- `static/js/main.js`: charts, filtering, search, modal previews, and HITL confirmations.
- `wine_classification_results/`: exported ML experiment artifacts.

## Folder Structure

```text
wine_classification_mvp/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── services/
├── templates/
├── static/
└── wine_classification_results/
    ├── models/
    ├── results/
    ├── figures/
    ├── processed_data/
    └── experiment_config.json
```

## Features

- Dashboard KPI cards from `results/final_summary.json`.
- Live wine classification using SentenceTransformer embeddings, `classifier.pkl` or `wine_classifier.pkl`, `label_encoder.pkl`, and OSR threshold configuration.
- Known prediction table with search and correctness badges.
- OSR metric cards, threshold display, decision table, and figure support.
- Semantic clustering metrics, Chart.js cluster distribution, figure display, dropdown filtering, and search.
- HITL review actions: approve, correct, reject.
- Verified labels are saved to `results/hitl_verified_labels.csv` without overwriting original experiment files.
- Retraining comparison table, chart, and figure display.
- Complete evaluation gallery with click-to-enlarge modal.
- Safe handling for missing optional files.

## Installation

1. Install Python 3.10+.
2. Open `wine_classification_mvp` in PyCharm.
3. Create or select a virtual environment.
4. Open the PyCharm terminal.
5. Run:

```bash
pip install -r requirements.txt
```

6. Put `wine_classification_results` inside the project root if it is not already there.
7. Run:

```bash
python app.py
```

8. Open:

```text
http://127.0.0.1:5000/
```

## How to Add ML Results

Place experiment files under `wine_classification_results`:

- Models in `models/`
- CSV and JSON reports in `results/`
- PNG/JPG figures in `figures/`
- prediction and clustering CSV files in `processed_data/`
- `experiment_config.json` at the root of `wine_classification_results/`

Optional files may be absent. The UI will display `Result not available for this experiment.` instead of crashing.

## Pages

- Dashboard
- Wine Classifier
- Open-Set Recognition
- Semantic Clustering
- HITL Labelling
- Model Retraining
- Results & Evaluation
- About Project

## Model Pipeline

User text is cleaned and converted to a BERT/SentenceTransformer embedding. The known-class classifier predicts class probabilities. The highest-confidence class is decoded through the label encoder. If confidence is below the OSR threshold, the record is shown as `UNKNOWN` with the closest known class retained for reference.

## HITL Workflow

The HITL page loads `results/hitl_labels.csv` or `results/HITL_review_required.csv`. Review decisions are written to `results/hitl_verified_labels.csv`, preserving the original review file. Statuses are `PENDING`, `APPROVED`, `CORRECTED`, and `REJECTED`.

## Troubleshooting

- If live prediction is unavailable, confirm the model files exist in `wine_classification_results/models/`.
- If SentenceTransformer download fails, connect to the internet once or use a locally cached model matching `experiment_config.json`.
- If a chart is empty, confirm the corresponding CSV contains numeric metric values.
- If a figure is missing, confirm the image exists under `wine_classification_results/figures/`.
- If port 5000 is already in use, stop the process using it or change the port in `app.py`.
