"""Experimental feedback: the 'validation -> feedback' step of the closed loop.

A submission is checked against the current classifier (agreement or discrepancy,
as in DigMethpy Step 3), saved as *pending*, and only enters the database after a
maintainer reviews it and moves it to curated_feedback.csv.

Storage note: on Streamlit Community Cloud the app's disk is reset on every
redeploy, so pending files written there are temporary. Submitters always get a
CSV download, and maintainers commit reviewed rows to data/feedback/.
"""

from datetime import datetime, timezone

import pandas as pd

FEEDBACK_COLUMNS = [
    "Feedback_ID",
    "Submitted_at_UTC",
    "Submitter",
    "Data_type",
    "LM_composition",
    "Base_LM",
    "Additive_metal_clean",
    "Reaction_type_clean",
    "Temp_representative_C",
    "Pressure_clean",
    "Voltage_V",
    "Current_density_mA_cm2",
    "Reaction_time_h_value",
    "Main_product",
    "Product_class",
    "CO2_conversion_representative_%",
    "FE_total_%",
    "Carbon_rate_umol_h",
    "Characterization_methods",
    "Source",
    "Model_predicted_class",
    "Model_probability",
    "Model_agreement",
    "Review_status",
]

PERCENT_FIELDS = ["CO2_conversion_representative_%", "FE_total_%"]


def validate_submission(entry):
    """Return a list of problems; empty means the entry can be saved."""
    problems = []
    for field in ["LM_composition", "Reaction_type_clean", "Main_product", "Product_class"]:
        if not str(entry.get(field) or "").strip():
            problems.append(f"'{field}' is required.")
    for field in PERCENT_FIELDS:
        v = entry.get(field)
        if v is not None and not (0 <= v <= 100):
            problems.append(f"'{field}' must be between 0 and 100.")
    t = entry.get("Temp_representative_C")
    if t is not None and not (-50 <= t <= 2000):
        problems.append("Temperature must be between -50 and 2000 °C.")
    return problems


def new_feedback_id(existing_count):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"FB-{stamp}-{existing_count + 1:03d}"


def as_row(entry):
    return {c: entry.get(c) for c in FEEDBACK_COLUMNS}


def append_pending(path, row):
    """Append to the pending file. Returns (ok, message); never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row], columns=FEEDBACK_COLUMNS)
        df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")
        return True, f"Saved to {path.name} on the app server (pending review)."
    except Exception as exc:  # read-only disk, permissions, ...
        return False, f"Could not write to the server ({exc}). Please download the CSV below instead."


def load_feedback(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def curated_as_database_rows(curated):
    """Curated feedback rows shaped like Analyzed_database rows, for the Database page."""
    if curated.empty:
        return curated
    out = curated.copy()
    out["Record_ID"] = out["Feedback_ID"]
    out["Paper_title"] = "Experimental feedback: " + out["Source"].fillna("unpublished").astype(str)
    out["Year"] = pd.to_datetime(out["Submitted_at_UTC"], errors="coerce").dt.year
    return out
