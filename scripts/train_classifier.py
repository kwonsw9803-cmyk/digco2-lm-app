"""Retrain the product-class classifier with the installed scikit-learn.

The pipeline and hyper-parameters are those of the original model (saved with scikit-learn 1.3.2);
it is fit on all 18 rows of Step4_ML_ready_dataset.xlsx, which reproduces that model exactly
under 1.3.2. Re-run this whenever the deployed scikit-learn version changes.

Usage (from the app folder): python scripts/train_classifier.py
"""

from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

APP = Path(__file__).resolve().parent.parent

CATEGORICAL = [
    "Reaction_type_for_ML",
    "Liquid_metal_clean_for_ML",
    "Additive_element_for_descriptor",
    "Additive_Element_group",
    "Additive_Common_oxidation_states",
    "Additive_Oxide_formula",
    "Additive_Oxide_formation_tendency",
    "Additive_Oxygen_affinity_class",
    "Additive_Reductant_strength_class",
    "Additive_CO2_to_solid_carbon_relevance",
    "Additive_Expected_role_in_LM_CO2",
]
NUMERIC = [
    "Temp_representative_C_ML",
    "Voltage_V_ML",
    "Current_density_mA_cm2_ML",
    "Is_electrochemical",
    "Is_thermal",
    "Is_mechanical",
    "Is_plasma",
    "Is_sonochemical",
    "Additive_Period",
    "Additive_Group_number",
    "Additive_Atomic_number",
    "Additive_Atomic_weight",
    "Additive_Melting_point_C",
    "Additive_Boiling_point_C",
    "Additive_Electronegativity_Pauling",
    "Additive_Atomic_radius_pm",
    "Additive_Standard_reduction_potential_V_vs_SHE",
]
TARGET = "Product_class"


def build():
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL), ("num", "passthrough", NUMERIC)]
    )
    clf = RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=42)
    return Pipeline([("preprocess", pre), ("classifier", clf)])


def main(out=APP / "product_classifier_random_forest.pkl"):
    data = pd.read_excel(APP / "Step4_ML_ready_dataset.xlsx", sheet_name="ML_ready_dataset")
    X, y = data[CATEGORICAL + NUMERIC], data[TARGET]
    model = build().fit(X, y)
    bundle = {
        "model": model,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "target_col": TARGET,
        "training_columns": CATEGORICAL + NUMERIC,
        "target_classes": sorted(y.unique()),
        "note": "Product_class classifier for LM-mediated CO2 conversion. Small literature dataset; proof-of-concept.",
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, out)
    print(f"saved {out.name} (scikit-learn {sklearn.__version__}, {len(X)} rows)")


if __name__ == "__main__":
    main()
