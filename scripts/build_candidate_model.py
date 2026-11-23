"""Build a 24-record candidate classifier NEXT TO the deployed one and report what would change.

Usage (from the app folder): python scripts/build_candidate_model.py

Writes only to data/review/candidate_model/ ; Step4_ML_ready_dataset.xlsx, Step6 and
product_classifier_random_forest.pkl are read, never modified.
The candidate uses the original recipe (recipe A of evaluate_classifier.py), only with 24 rows.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_classifier as ev  # noqa: E402
import train_classifier as tc  # noqa: E402

OUT = tc.APP / "data" / "review" / "candidate_model"
SC = "solid carbon"


def p_solid(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(SC)]


def main():
    df = ev.load_rows()
    bad = ev.check_step4(df)
    if bad:
        raise SystemExit(f"first 18 rows differ from Step4: {bad}")
    data = ev.prepare(df, "A").drop(columns="Group")
    cols = tc.CATEGORICAL + tc.NUMERIC
    model = tc.build().fit(data[cols], data[tc.TARGET])
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "ML_ready_dataset_24.csv", index=False, encoding="utf-8-sig")
    bundle = {
        "model": model,
        "categorical_features": tc.CATEGORICAL,
        "numeric_features": tc.NUMERIC,
        "target_col": tc.TARGET,
        "training_columns": cols,
        "target_classes": sorted(data[tc.TARGET].unique()),
        "note": "CANDIDATE, not deployed: 24 literature records. Does not beat the majority-class baseline.",
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, OUT / "product_classifier_candidate_24.pkl")
    num = ["Temp_representative_C_ML", "Voltage_V_ML", "Current_density_mA_cm2_ML"]
    print("reported values per column (of 24):", df[num].notna().sum().to_dict(), "| medians:", df[num].median().to_dict())

    old = joblib.load(tc.APP / "product_classifier_random_forest.pkl")["model"]
    for sheet in ["All_candidate_predictions", "Top_recommendations"]:
        s6 = pd.read_excel(tc.APP / "Step6_agent_recommendation_results.xlsx", sheet_name=sheet)
        X = s6[cols].copy()
        X[tc.CATEGORICAL] = X[tc.CATEGORICAL].fillna("Not reported").astype(str)
        X[tc.NUMERIC] = X[tc.NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0)
        po, pn = old.predict(X), model.predict(X)
        qo, qn = p_solid(old, X), p_solid(model, X)
        print(f"\n{sheet}: n={len(X)}")
        print("  deployed model reproduces the stored class:", (po == s6["Predicted_Product_class"]).mean())
        print("  max |stored ML_target_probability - deployed P(solid carbon)|:", abs(s6["ML_target_probability"] - qo).max())
        print("  class changes with the candidate:", int((po != pn).sum()))
        print(f"  mean P(solid carbon) {qo.mean():.3f} -> {qn.mean():.3f}, max |change| {abs(qn - qo).max():.3f}")
        print(pd.crosstab(pd.Series(po, name="deployed"), pd.Series(pn, name="candidate")).to_string())


if __name__ == "__main__":
    main()
