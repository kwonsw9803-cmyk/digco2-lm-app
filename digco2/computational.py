"""Computational (DFT / AIMD) records and descriptor regression.

Records are stored in long format, one computed property per row, so new
properties need no schema change. Regression follows the DigMethpy idea
(Fig. 4A): predict a physically meaningful descriptor of the solute (e.g. its
Bader charge or CO2 adsorption energy) from elemental features, evaluated with
leave-one-out CV and compared with a predict-the-mean baseline.
"""

import numpy as np
import pandas as pd

COMP_COLUMNS = [
    "Record_ID",
    "System",
    "Base_LM",
    "Solute",
    "Solute_at_pct",
    "Temperature_K",
    "Method",
    "Functional_or_potential",
    "Property",
    "Value",
    "Unit",
    "Notes",
    "Source",
]

SUGGESTED_PROPERTIES = {
    "Bader_charge_solute": "e (average over solute atoms; negative = electron-rich)",
    "E_ads_CO2": "eV (CO2 adsorption energy on the molten surface)",
    "E_ads_O": "eV (O adsorption energy; oxygen-scavenging strength)",
    "CO2_activation_barrier": "eV (C-O bond-breaking barrier)",
    "Diffusion_coefficient_solute": "cm2/s (from AIMD mean-squared displacement)",
    "Surface_segregation_energy": "eV (negative = solute enriches at surface)",
    # bulk thermodynamic descriptors already provided from OQMD
    "Oxide_formation_energy_per_O": "eV/O atom (bulk oxide; more negative = stronger oxygen scavenger)",
    "Carbide_formation_energy": "eV/atom (bulk carbide; 0 = no stable carbide)",
    "Ga_X_compound_formation_energy": "eV/atom (bulk Ga–X compound; 0 = no stable compound)",
}

TEMPLATE_ROWS = [
    {
        "Record_ID": "C-0001",
        "System": "Ga46Zn4 (example)",
        "Base_LM": "Ga",
        "Solute": "Zn",
        "Solute_at_pct": 8.0,
        "Temperature_K": 923,
        "Method": "AIMD + DFT",
        "Functional_or_potential": "PBE-D3",
        "Property": "Bader_charge_solute",
        "Value": None,
        "Unit": "e",
        "Notes": "Example row: replace with your own values",
        "Source": "your DOI or internal ID",
    }
]


def template_csv():
    return pd.DataFrame(TEMPLATE_ROWS, columns=COMP_COLUMNS).to_csv(index=False, encoding="utf-8-sig")


def load_computational(path):
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=COMP_COLUMNS)
    return df


def validate_computational(df):
    """Return (clean_df, issues). Rows without Solute, Property or a numeric Value are dropped."""
    issues = []
    missing = [c for c in ["Solute", "Property", "Value"] if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=COMP_COLUMNS), [f"Missing required columns: {', '.join(missing)}"]
    df = df.copy()
    for c in COMP_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    bad = df["Value"].isna() | df["Solute"].isna() | df["Property"].isna()
    if bad.any():
        issues.append(f"{int(bad.sum())} row(s) skipped: Solute, Property and a numeric Value are required.")
    df = df[~bad]
    df["Solute"] = df["Solute"].astype(str).str.strip()
    df["Property"] = df["Property"].astype(str).str.strip()
    return df[COMP_COLUMNS], issues


def regression_table(comp, desc, prop, features):
    """Mean property value per solute, joined to that solute's elemental features."""
    sub = comp[comp["Property"] == prop]
    target = sub.groupby("Solute")["Value"].agg(["mean", "count"]).reset_index()
    target.columns = ["Element", "Target", "n_records"]
    table = target.merge(desc[["Element"] + features], on="Element", how="left")
    unknown = sorted(table.loc[table[features].isna().any(axis=1), "Element"])
    return table.dropna(subset=features), unknown


def loocv_regression(table, features, model_name="Ridge"):
    """Leave-one-out predictions and metrics, plus a predict-the-training-mean baseline."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = table[features].astype(float).to_numpy()
    y = table["Target"].astype(float).to_numpy()

    if model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators=300, random_state=0, min_samples_leaf=1)
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    loo = LeaveOneOut()
    pred = cross_val_predict(model, X, y, cv=loo)
    base = np.array([np.delete(y, i).mean() for i in range(len(y))])

    def metrics(p):
        err = p - y
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "MAE": float(np.mean(np.abs(err))),
            "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }

    result = table[["Element", "Target"]].copy()
    result["LOOCV_prediction"] = pred
    return result, {"model": metrics(pred), "baseline": metrics(base)}
