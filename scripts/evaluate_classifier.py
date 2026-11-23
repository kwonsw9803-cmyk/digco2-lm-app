"""Compare product-class classifiers on the 18 original and the 24 current literature records.

Usage (from the app folder): python scripts/evaluate_classifier.py [out.json]

Nothing is deployed: the script reads the workbooks and the current model and writes only the JSON report.
The app shows the report saved as data/model_evaluation.json. Takes about half an hour.

Recipes (fixed before looking at any score):
  A  original recipe: 28 features, missing values filled with medians over all rows (as in Step4)
  B  A without voltage and current density (mixed reference electrodes, mostly imputed); medians are
     fitted inside each training fold
  C  B without the liquid-metal name (nearly unique per row) and the Is_* flags (duplicate Reaction_type)
All use the same random forest as the original model. Baseline: always the majority class of the training fold.
Cross-validation: leave-one-record-out, and leave-one-paper-out (records 15-18 are one thesis).
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_classifier as tc  # noqa: E402

APP = tc.APP
FINAL = APP / "Final_LM_CO2_platform_database_with_descriptors.xlsx"
STEP4 = APP / "Step4_ML_ready_dataset.xlsx"
SEEDS = range(10)
NR = "Not reported"
THESIS = {15, 16, 17, 18}
IS_FLAGS = [c for c in tc.NUMERIC if c.startswith("Is_")]
VOLT = ["Voltage_V_ML", "Current_density_mA_cm2_ML"]
DESC_CAT = [c for c in tc.CATEGORICAL if c.startswith("Additive_") and c != "Additive_element_for_descriptor"]
DESC_NUM = [c for c in tc.NUMERIC if c.startswith("Additive_")]


def flags(rt):
    """Same rule as the app and Step4 ('chemical' also matches electrochemical / sonochemical)."""
    return {
        "Is_electrochemical": int("electrochemical" in rt),
        "Is_thermal": int("thermal" in rt or "chemical" in rt),
        "Is_mechanical": int("mechanical" in rt),
        "Is_plasma": int("plasma" in rt),
        "Is_sonochemical": int("sono" in rt),
    }


def load_rows():
    """24 rows with the original encodings; unreported numbers stay NaN."""
    norm = pd.read_excel(FINAL, sheet_name="Normalized_database")
    desc = pd.read_excel(FINAL, sheet_name="Element_descriptors").set_index("Element")
    raw = pd.read_excel(STEP4, sheet_name="Descriptor_merged_raw")
    known_el = dict(zip(raw["Record_ID"], raw["Additive_element_for_descriptor"]))
    known_lm = dict(zip(raw["Record_ID"], raw["Liquid_metal_clean_for_ML"]))
    rows = []
    for _, r in norm.iterrows():
        rid, rt = int(r["Record_ID"]), r["Reaction_type_clean"]
        el = known_el.get(rid, r["Additive_metal_clean"])
        row = {
            "Record_ID": rid,
            "Group": "thesis" if rid in THESIS else f"paper{rid}",
            "Reaction_type_for_ML": rt,
            "Liquid_metal_clean_for_ML": known_lm.get(rid, r["LM_composition"]),
            "Additive_element_for_descriptor": el,
            "Temp_representative_C_ML": pd.to_numeric(r["Temp_representative_C"], errors="coerce"),
            "Voltage_V_ML": pd.to_numeric(r["Voltage_V"], errors="coerce"),
            "Current_density_mA_cm2_ML": pd.to_numeric(r["Current_density_mA_cm2"], errors="coerce"),
            "Product_class": r["Product_class"],
            **flags(rt),
        }
        for c in DESC_CAT + DESC_NUM:
            src = c[len("Additive_"):]
            row[c] = desc.at[el, src] if el in desc.index else (NR if c in DESC_CAT else np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df[DESC_CAT] = df[DESC_CAT].fillna(NR).astype(str)
    df[tc.NUMERIC] = df[tc.NUMERIC].apply(pd.to_numeric, errors="coerce")
    return df


def check_step4(df):
    """The first 18 rows, median-filled, must equal the stored ML dataset."""
    stored = pd.read_excel(STEP4, sheet_name="ML_ready_dataset")
    mine = df.iloc[:18].copy()
    mine[tc.NUMERIC] = mine[tc.NUMERIC].fillna(mine[tc.NUMERIC].median())
    bad = [c for c in tc.CATEGORICAL + ["Product_class"] if list(mine[c].astype(str)) != list(stored[c].astype(str))]
    bad += [c for c in tc.NUMERIC if not np.allclose(mine[c], stored[c])]
    return bad


def features(recipe):
    cat, num = list(tc.CATEGORICAL), list(tc.NUMERIC)
    if recipe in ("B", "C"):
        num = [c for c in num if c not in VOLT]
    if recipe == "C":
        cat.remove("Liquid_metal_clean_for_ML")
        num = [c for c in num if c not in IS_FLAGS]
    return cat, num


def make(recipe, seed=42):
    if recipe == "baseline":
        return DummyClassifier(strategy="most_frequent")
    cat, num = features(recipe)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), num),
    ])
    clf = RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=seed)
    return Pipeline([("preprocess", pre), ("classifier", clf)])


def prepare(df, recipe):
    d = df.copy()
    if recipe == "A":  # original behaviour: medians over every row, before any split
        d[tc.NUMERIC] = d[tc.NUMERIC].fillna(d[tc.NUMERIC].median())
    return d


def cv_predict(df, y, groups, recipe, seed):
    d = prepare(df, recipe)
    cols = sum(features(recipe if recipe != "baseline" else "A"), [])
    pred, proba = np.empty(len(d), dtype=object), np.full(len(d), np.nan)
    for tr, te in LeaveOneGroupOut().split(d, y, groups):
        m = make(recipe, seed).fit(d.iloc[tr][cols], y[tr])
        pred[te] = m.predict(d.iloc[te][cols])
        if "solid carbon formed" in m.classes_:
            proba[te] = m.predict_proba(d.iloc[te][cols])[:, list(m.classes_).index("solid carbon formed")]
    return pred, proba


def wilson(k, n, z=1.96):
    p, den = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(mid - half), 3), round(float(mid + half), 3)]


def score(df, y, groups, recipe, binary=False):
    out = []
    for s in ([0] if recipe == "baseline" else SEEDS):
        pred, proba = cv_predict(df, y, groups, recipe, s)
        out.append({
            "correct": int((pred == y).sum()),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "auc": roc_auc_score(y == "solid carbon formed", proba) if binary and recipe != "baseline" else np.nan,
        })
    r = pd.DataFrame(out)
    k = r["correct"].median()
    return {
        "n": len(y), "correct_median": float(k), "correct_range": [int(r["correct"].min()), int(r["correct"].max())],
        "accuracy": round(float(r["correct"].mean() / len(y)), 3), "wilson95_of_median": wilson(k, len(y)),
        "balanced_accuracy": round(float(r["balanced_accuracy"].mean()), 3),
        "macro_f1": round(float(r["macro_f1"].mean()), 3),
        "auc": None if r["auc"].isna().all() else round(float(r["auc"].mean()), 3),
    }


def loocv_published_settings():
    """The LOOCV accuracy stored in Step4 Model_summary vs. the settings the deployed model really uses."""
    d = pd.read_excel(STEP4, sheet_name="ML_ready_dataset")
    X, y = d[tc.CATEGORICAL + tc.NUMERIC], d[tc.TARGET]
    res = {}
    for name, cw in [("deployed_settings_balanced", "balanced"), ("without_class_weight", None)]:
        m = tc.build().set_params(classifier__class_weight=cw)
        res[name] = round(float(cross_val_score(m, X, y, cv=LeaveOneOut()).mean()), 3)
    return res


def step6_if_retrained(df):
    """What the stored Step 6 ranking would become with the 24-record model (recipe A, not deployed)."""
    cols = tc.CATEGORICAL + tc.NUMERIC
    d = prepare(df, "A")
    new = tc.build().fit(d[cols], d[tc.TARGET])
    s6 = pd.read_excel(APP / "Step6_agent_recommendation_results.xlsx", sheet_name="All_candidate_predictions")
    X = s6[cols].copy()
    X[tc.CATEGORICAL] = X[tc.CATEGORICAL].fillna(NR).astype(str)
    p = new.predict_proba(X)[:, list(new.classes_).index("solid carbon")]
    before = s6["Final_recommendation_score"]
    rule = s6["Rule_based_score"]
    if not np.allclose(before, 10 * s6["ML_target_probability"] + rule):
        raise SystemExit("Step6 score is not 10 * ML probability + rule score")
    top0, top1 = before.nlargest(20).index, (10 * p + rule).nlargest(20).index
    el = "Additive_element_for_descriptor"
    return {
        "candidates": len(s6),
        "class_changes": int((s6["Predicted_Product_class"] != new.predict(X)).sum()),
        "mean_p_solid_carbon": [round(float(s6["ML_target_probability"].mean()), 3), round(float(p.mean()), 3)],
        "top20_overlap": len(set(top0) & set(top1)),
        "top20_additives": [sorted(s6.loc[top0, el].unique()), sorted(s6.loc[top1, el].unique())],
    }


def main(out=None):
    warnings.filterwarnings("ignore")
    df = load_rows()
    bad = check_step4(df)
    print("first 18 rows equal Step4 ML_ready_dataset:", "yes" if not bad else f"NO -> {bad}")
    if bad:
        raise SystemExit(1)
    y = df["Product_class"].to_numpy(dtype=object)
    report = {"class_counts_24": df["Product_class"].value_counts().to_dict()}
    new = df["Record_ID"] > 18

    # External test: the deployed model has never seen records 19-24.
    bundle = joblib.load(APP / "product_classifier_random_forest.pkl")
    med18 = df.loc[~new, tc.NUMERIC].median()
    X_new = df.loc[new, bundle["training_columns"]].copy()
    X_new[tc.NUMERIC] = X_new[tc.NUMERIC].fillna(med18)
    p = bundle["model"].predict(X_new)
    report["deployed_on_new_records"] = [
        {"Record_ID": int(i), "actual": a, "predicted": q} for i, a, q in zip(df.loc[new, "Record_ID"], y[new], p)]
    print("\n== deployed model on the 6 unseen records:", int((p == y[new]).sum()), "of 6 correct")
    for r in report["deployed_on_new_records"]:
        print("  ", r)

    schemes = {"leave-one-record-out": df["Record_ID"].to_numpy(), "leave-one-paper-out": df["Group"].to_numpy()}
    for label, mask in [("18 original records", ~new.to_numpy()), ("24 records", np.ones(len(df), bool))]:
        for scheme, g in schemes.items():
            print(f"\n== {label}, {scheme}")
            for rec in ["baseline", "A", "B", "C"]:
                s = score(df[mask].reset_index(drop=True), y[mask], g[mask], rec)
                report[f"{label} | {scheme} | {rec}"] = s
                print(f"   {rec:9s} {s}")

    yb = np.where(df["Product_class"].str.startswith("solid carbon"), "solid carbon formed", "other").astype(object)
    report["binary_counts_24"] = pd.Series(yb).value_counts().to_dict()
    print("\n== 24 records, binary target (solid carbon formed vs other), leave-one-paper-out", report["binary_counts_24"])
    for rec in ["baseline", "A", "B", "C"]:
        s = score(df, yb, schemes["leave-one-paper-out"], rec, binary=True)
        report[f"binary | leave-one-paper-out | {rec}"] = s
        print(f"   {rec:9s} {s}")

    rule = np.where(df["Reaction_type_for_ML"] == "electrochemical", "other", "solid carbon formed")
    report["binary_single_rule_correct"] = int((rule == yb).sum())  # read off the same data: a sanity check, not a model
    report["loocv_18_published_settings"] = loocv_published_settings()
    report["step6_if_retrained"] = step6_if_retrained(df)
    report["inputs"] = {
        "reported_voltage": int(df["Voltage_V_ML"].notna().sum()),
        "reported_current_density": int(df["Current_density_mA_cm2_ML"].notna().sum()),
        "reported_temperature": int(df["Temp_representative_C_ML"].notna().sum()),
        "distinct_liquid_metal_names": int(df["Liquid_metal_clean_for_ML"].nunique()),
        "records": len(df),
    }
    for k in ["binary_single_rule_correct", "loocv_18_published_settings", "step6_if_retrained", "inputs"]:
        print(f"\n== {k}: {report[k]}")

    if out:
        Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:2])
