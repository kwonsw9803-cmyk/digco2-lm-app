"""Carry the fitted RandomForest over to the installed scikit-learn without changing its predictions.

The original pickle (scikit-learn 1.3.2) loads under newer versions, but its ColumnTransformer no longer
runs. The fitted trees still do, so this keeps the original classifier and refits only the deterministic
preprocessing (OneHotEncoder + passthrough) on the same 18 training rows, then re-saves the bundle.
Re-training from scratch (train_classifier.py) gives a different forest because RNG use changed between
versions; migrating keeps the Step 6 / Step 7 results consistent with the model in the app.

Usage (from the app folder): python scripts/migrate_model.py [old.pkl] [new.pkl]
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_classifier as tc  # noqa: E402

APP = tc.APP


def main(old_path=APP / "product_classifier_random_forest.pkl", new_path=APP / "product_classifier_random_forest.pkl"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # InconsistentVersionWarning from the old pickle
        old = joblib.load(old_path)
    old_model = old["model"]
    old_clf = old_model.named_steps["classifier"]
    old_ohe = old_model.named_steps["preprocess"].named_transformers_["cat"]

    data = pd.read_excel(APP / "Step4_ML_ready_dataset.xlsx", sheet_name="ML_ready_dataset")
    X = data[old["training_columns"]]
    pre = tc.build().named_steps["preprocess"].fit(X)
    new_ohe = pre.named_transformers_["cat"]
    if len(old_ohe.categories_) != len(new_ohe.categories_) or any(
        list(a) != list(b) for a, b in zip(old_ohe.categories_, new_ohe.categories_)
    ):
        raise SystemExit("One-hot categories differ from the original model; not migrating.")

    # scikit-learn >= 1.4 stores classifier leaf values as class fractions; 1.3 stored weighted counts.
    # Normalising each node in place gives the same predict_proba as the original model.
    for est in old_clf.estimators_:
        v = est.tree_.value
        sums = v.sum(axis=2, keepdims=True)
        if not np.allclose(sums, 1.0):
            v /= np.where(sums > 0, sums, 1.0)
        if not np.allclose(est.tree_.value.sum(axis=2), 1.0):
            raise SystemExit("Could not normalise tree values in place; not migrating.")

    model = Pipeline([("preprocess", pre), ("classifier", old_clf)])
    model.predict_proba(X)  # must run under the installed version
    bundle = dict(old, model=model, sklearn_version=sklearn.__version__, migrated_from="scikit-learn 1.3.2 pickle")
    joblib.dump(bundle, new_path)
    print(f"saved {Path(new_path).name} under scikit-learn {sklearn.__version__}")
    return model


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:3]]
    main(*args)
