import json
from io import BytesIO
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for

from config import FIGURES_DIR, MAX_PREDICTION_TEXT_LENGTH, SECRET_KEY
from services import data_service
from services.hitl_service import export_verified_labels, load_review_records, review_stats, update_review_status
from services.model_service import live_prediction_status, predict_wine


app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY


def records(df, limit=None):
    return data_service.dataframe_records(df, limit=limit)


def metrics_for_chart(metric_map, candidates):
    labels = []
    values = []
    for label, keys in candidates:
        value = data_service.find_metric(metric_map, keys)
        try:
            if value is not None:
                labels.append(label)
                values.append(round(float(value), 4))
        except (TypeError, ValueError):
            continue
    return {"labels": labels, "values": values}


@app.context_processor
def inject_helpers():
    return {
        "missing_message": data_service.MISSING_MESSAGE,
        "fmt": data_service.format_value,
        "figure_url": lambda filename: url_for("figure", filename=filename) if filename else "",
    }


@app.route("/")
def dashboard():
    summary = data_service.get_final_summary()
    classification_map = data_service.metric_map(data_service.get_classification_metrics())
    osr_map = data_service.metric_map(data_service.get_osr_metrics())
    retraining_map = data_service.metric_map(data_service.get_retraining_results())
    kpis = [
        ("Total Dataset Records", data_service.find_metric(summary, ["full_clean_dataset_records", "total_dataset_records"])),
        ("Unique Wine Keys", data_service.find_metric(summary, ["full_unique_wine_keys", "unique_wine_keys"])),
        ("Known Classes", data_service.find_metric(summary, ["known_classes"])),
        ("Held-Out Unknown Classes", data_service.find_metric(summary, ["heldout_unknown_classes", "held_out_unknown_classes"])),
        ("Training Records", data_service.find_metric(summary, ["training_records"])),
        ("Classification Accuracy", data_service.find_metric(summary, ["classification_accuracy"])),
        ("Classification F1", data_service.find_metric(summary, ["classification_f1"])),
        ("OSR Unknown Detection Rate", data_service.find_metric(summary, ["unknown_detection_rate", "osr_unknown_detection_rate"])),
        ("OSR AUROC", data_service.find_metric(summary, ["osr_auroc", "auroc"])),
        ("Semantic Clusters", data_service.find_metric(summary, ["semantic_clusters"])),
        ("Records Clustered", data_service.find_metric(summary, ["records_semantically_clustered", "records_clustered"])),
        ("Retrained Model Accuracy", data_service.find_metric(summary, ["new_class_accuracy_after_retraining", "known_accuracy_after_retraining", "retrained_model_accuracy"])),
    ]
    return render_template(
        "dashboard.html",
        title="Dashboard",
        summary=summary,
        kpis=kpis,
        classification_chart=metrics_for_chart(classification_map, [
            ("Accuracy", ["Accuracy", "classification_accuracy"]),
            ("Precision", ["Precision", "classification_precision"]),
            ("Recall", ["Recall", "classification_recall"]),
            ("F1", ["F1-score", "F1", "classification_f1"]),
        ]),
        osr_chart=metrics_for_chart(osr_map, [
            ("Accuracy", ["OSR Accuracy", "osr_accuracy"]),
            ("Precision", ["Unknown Precision", "unknown_precision"]),
            ("Recall", ["Unknown Recall", "unknown_recall"]),
            ("F1", ["Unknown F1", "unknown_f1"]),
            ("AUROC", ["AUROC", "osr_auroc"]),
        ]),
        retraining_chart=metrics_for_chart(retraining_map, [
            ("Before", ["Before Retraining", "before_accuracy", "Original Accuracy"]),
            ("After", ["After Retraining", "after_accuracy", "Retrained Accuracy"]),
        ]),
    )


@app.route("/classifier")
def classifier():
    df = data_service.get_known_predictions()
    rows = records(df, 500)
    stats = {"total": 0, "correct": 0, "incorrect": 0, "avg_confidence": None}
    if df is not None and not df.empty:
        actual = "actual_label" if "actual_label" in df.columns else "winekey" if "winekey" in df.columns else None
        predicted = "predicted_label" if "predicted_label" in df.columns else None
        stats["total"] = int(len(df))
        if actual and predicted:
            correct = (df[actual].astype(str) == df[predicted].astype(str)).sum()
            stats["correct"] = int(correct)
            stats["incorrect"] = int(len(df) - correct)
        if "confidence" in df.columns:
            stats["avg_confidence"] = df["confidence"].mean()
    return render_template(
        "classifier.html",
        title="Wine Classifier",
        prediction_status=live_prediction_status(),
        rows=rows,
        stats=stats,
        confusion_matrix=data_service.safe_image("confusion_matrix.png"),
        classification_metrics_img=data_service.safe_image("classification_metrics.png"),
    )


@app.route("/classifier/predict", methods=["POST"])
def classifier_predict():
    text = (request.form.get("wine_text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter a wine name or description."}), 400
    if len(text) > MAX_PREDICTION_TEXT_LENGTH:
        return jsonify({"error": f"Input is too long. Limit is {MAX_PREDICTION_TEXT_LENGTH} characters."}), 400
    result = predict_wine(text)
    status = 400 if "error" in result else 200
    return jsonify(result), status


@app.route("/osr")
def osr():
    osr_metrics = data_service.metric_map(data_service.get_osr_metrics())
    osr_config = data_service.get_osr_config()
    unknown_df = data_service.get_unknown_predictions()
    return render_template(
        "osr.html",
        title="Open-Set Recognition",
        metrics=osr_metrics,
        threshold=data_service.find_metric(osr_config, ["threshold", "osr_threshold", "confidence_threshold", "optimal_threshold"]),
        rows=records(unknown_df, 800),
        osr_analysis=data_service.safe_image("osr_analysis.png"),
        osr_confidence=data_service.safe_image("osr_confidence_distribution.png"),
        osr_roc=data_service.safe_image("osr_roc_curve.png"),
    )


@app.route("/clustering")
def clustering():
    df = data_service.get_clustered_records()
    metrics = data_service.metric_map(data_service.get_clustering_metrics())
    cluster_counts = []
    cluster_rows = []
    stats = {"total": 0, "clusters": 0, "largest": "N/A", "smallest": "N/A"}
    if df is not None and not df.empty and "cluster" in df.columns:
        counts = df["cluster"].value_counts().sort_index()
        cluster_counts = [{"cluster": str(k), "count": int(v)} for k, v in counts.items()]
        stats = {
            "total": int(len(df)),
            "clusters": int(counts.shape[0]),
            "largest": int(counts.max()),
            "smallest": int(counts.min()),
        }
        for cluster_id, group in df.groupby("cluster"):
            cluster_rows.append({
                "cluster": str(cluster_id),
                "count": int(len(group)),
                "samples": " | ".join(group.get("text", group.iloc[:, 0]).astype(str).head(3).tolist()),
                "source": ", ".join(sorted(set(group.get("source", "").astype(str)))) if "source" in group.columns else "",
            })
    return render_template(
        "clustering.html",
        title="Semantic Clustering",
        metrics=metrics,
        stats=stats,
        cluster_counts=cluster_counts,
        cluster_rows=cluster_rows,
        cluster_visualization=data_service.safe_image(["cluster_visualization.png", "semantic_clusters.png"]),
    )


@app.route("/hitl")
def hitl():
    df = load_review_records()
    return render_template("hitl.html", title="HITL Labelling", rows=records(df, 1000), stats=review_stats(df))


@app.route("/hitl/update", methods=["POST"])
def hitl_update():
    record_id = (request.form.get("record_id") or "").strip()
    action = (request.form.get("action") or "").strip()
    corrected_label = (request.form.get("corrected_label") or "").strip()
    ok, message = update_review_status(record_id, action, corrected_label)
    flash(message, "success" if ok else "danger")
    return redirect(url_for("hitl"))


@app.route("/retraining")
def retraining():
    df = data_service.get_retraining_results()
    summary = data_service.get_final_summary()
    metric_values = data_service.metric_map(df)
    return render_template(
        "retraining.html",
        title="Model Retraining",
        rows=records(df),
        summary=summary,
        chart=metrics_for_chart(metric_values, [
            ("Before Accuracy", ["Before Accuracy", "Original Accuracy", "before_accuracy"]),
            ("After Accuracy", ["After Accuracy", "Retrained Accuracy", "after_accuracy"]),
            ("Before F1", ["Before F1", "Original F1", "before_f1"]),
            ("After F1", ["After F1", "Retrained F1", "after_f1"]),
        ]),
        retraining_image=data_service.safe_image(["retraining_comparison.png", "retraining_performance_comparison.png"]),
    )


@app.route("/results")
def results():
    summary = data_service.get_final_summary()
    sections = {
        "Known Classification": records(data_service.get_classification_metrics()),
        "Open-Set Recognition": records(data_service.get_osr_metrics()),
        "Semantic Clustering": records(data_service.get_clustering_metrics()),
        "Retraining": records(data_service.get_retraining_results()),
    }
    hitl_df = load_review_records()
    return render_template(
        "results.html",
        title="Results & Evaluation",
        summary=summary,
        sections=sections,
        hitl_stats=review_stats(hitl_df),
        figures=data_service.available_figures(),
    )


@app.route("/about")
def about():
    return render_template("about.html", title="About Project")


@app.route("/download/hitl")
def download_hitl():
    path = export_verified_labels()
    return send_file(path, as_attachment=True, download_name="hitl_verified_labels.csv")


@app.route("/download/summary")
def download_summary():
    summary = data_service.get_final_summary()
    payload = json.dumps(summary, indent=2).encode("utf-8")
    return send_file(BytesIO(payload), as_attachment=True, download_name="experiment_summary.json", mimetype="application/json")


@app.route("/figures/<path:filename>")
def figure(filename):
    safe_name = Path(filename).name
    path = FIGURES_DIR / safe_name
    if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return data_service.MISSING_MESSAGE, 404
    return send_from_directory(FIGURES_DIR, safe_name)


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", title="Page Not Found", code=404, message="The requested page was not found."), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("error.html", title="Application Error", code=500, message="The application could not complete the request."), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
