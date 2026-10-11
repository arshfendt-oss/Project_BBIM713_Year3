from pathlib import Path
from typing import Any

import pandas as pd

from config import HITL_VERIFIED_FILENAME, REPORTS_DIR
from services.data_service import get_hitl_records, safe_read_csv


VERIFIED_PATH = REPORTS_DIR / HITL_VERIFIED_FILENAME
ALLOWED_ACTIONS = {"approve", "correct", "reject"}
STATUS_FOR_ACTION = {
    "approve": "APPROVED",
    "correct": "CORRECTED",
    "reject": "REJECTED",
}


def _record_id(row: pd.Series, index: int) -> str:
    for col in ("record_id", "id", "Record ID"):
        if col in row and str(row[col]).strip():
            return str(row[col])
    return str(index + 1)


def _normalize_review_records(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=[
            "record_id", "text", "cluster", "suggested_cluster_label",
            "human_verified_label", "verification_status", "source",
        ])
    data = df.copy().fillna("")
    if "record_id" not in data.columns:
        data.insert(0, "record_id", [_record_id(row, i) for i, row in data.iterrows()])
    if "verification_status" not in data.columns:
        data["verification_status"] = "PENDING"
    if "human_verified_label" not in data.columns:
        data["human_verified_label"] = ""
    if "suggested_cluster_label" not in data.columns:
        if "suggested_label" in data.columns:
            data["suggested_cluster_label"] = data["suggested_label"]
        elif "cluster" in data.columns:
            data["suggested_cluster_label"] = data["cluster"].apply(lambda x: f"UNKNOWN_CLUSTER_{x}")
        else:
            data["suggested_cluster_label"] = ""
    if "cluster" not in data.columns:
        data["cluster"] = ""
    if "source" not in data.columns:
        data["source"] = ""
    return data


def _merge_verified(base: pd.DataFrame, verified: pd.DataFrame | None) -> pd.DataFrame:
    if verified is None or verified.empty:
        return base
    merged = base.copy()
    verified = verified.drop_duplicates(subset=["record_id"], keep="last")
    updates = verified.set_index("record_id")
    for idx, row in merged.iterrows():
        record_id = str(row["record_id"])
        if record_id in updates.index:
            update = updates.loc[record_id]
            for column in ("human_verified_label", "verification_status", "review_action"):
                if column in update.index:
                    merged.at[idx, column] = update[column]
    return merged


def load_review_records() -> pd.DataFrame:
    base = _normalize_review_records(get_hitl_records())
    verified = safe_read_csv(VERIFIED_PATH)
    return _merge_verified(base, verified).fillna("")


def save_verified_record(record: dict[str, Any]) -> None:
    VERIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = safe_read_csv(VERIFIED_PATH)
    new_row = pd.DataFrame([record])
    if existing is None or existing.empty:
        combined = new_row
    else:
        combined = pd.concat([existing, new_row], ignore_index=True)
        combined = combined.drop_duplicates(subset=["record_id"], keep="last")
    combined.to_csv(VERIFIED_PATH, index=False)


def update_review_status(record_id: str, action: str, corrected_label: str = "") -> tuple[bool, str]:
    action = (action or "").lower().strip()
    if action not in ALLOWED_ACTIONS:
        return False, "Invalid HITL action."

    records = load_review_records()
    if records.empty or record_id not in set(records["record_id"].astype(str)):
        return False, "Review record was not found."

    row = records[records["record_id"].astype(str) == str(record_id)].iloc[0].to_dict()
    suggested = str(row.get("suggested_cluster_label", ""))
    status = STATUS_FOR_ACTION[action]
    if action == "correct":
        label = corrected_label.strip()
        if not label:
            return False, "A corrected label is required."
    elif action == "approve":
        label = suggested
    else:
        label = ""

    save_verified_record({
        "record_id": str(record_id),
        "text": row.get("text", row.get("wine_description", "")),
        "cluster": row.get("cluster", ""),
        "suggested_cluster_label": suggested,
        "human_verified_label": label,
        "verification_status": status,
        "review_action": action.upper(),
        "source": row.get("source", ""),
    })
    return True, f"Record {record_id} marked as {status}."


def review_stats(df: pd.DataFrame | None = None) -> dict[str, Any]:
    records = load_review_records() if df is None else df
    total = int(len(records))
    statuses = records.get("verification_status", pd.Series(dtype=str)).fillna("PENDING").astype(str).str.upper()
    pending = int((statuses == "PENDING").sum())
    approved = int((statuses == "APPROVED").sum())
    corrected = int((statuses == "CORRECTED").sum())
    rejected = int((statuses == "REJECTED").sum())
    completed = approved + corrected + rejected
    completion = round((completed / total * 100), 2) if total else 0
    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "corrected": corrected,
        "rejected": rejected,
        "completion": completion,
    }


def export_verified_labels() -> Path:
    if not VERIFIED_PATH.exists():
        empty = pd.DataFrame(columns=[
            "record_id", "text", "cluster", "suggested_cluster_label",
            "human_verified_label", "verification_status", "review_action", "source",
        ])
        VERIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(VERIFIED_PATH, index=False)
    return VERIFIED_PATH
