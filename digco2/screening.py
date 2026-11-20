"""Descriptor-based screening of additive elements (DigMethpy Fig. 4B analogue).

Each element in the descriptor table is tagged with what the literature database
reports for it, then placed on a two-descriptor map. Untested elements are ranked
by how close they sit, in standardized descriptor space, to elements that already
gave the target product. This is a transparent similarity heuristic, not a model.
"""

import re

import numpy as np
import pandas as pd

BASE_LM_ELEMENTS = {"Ga"}

STATUS_TARGET = "Target product reported"
STATUS_OTHER = "Other products reported"
STATUS_UNTESTED = "Not yet tested"
STATUS_BASE = "Base liquid metal"

NUMERIC_DESCRIPTORS = {
    "Electronegativity_Pauling": "Electronegativity (Pauling)",
    "Standard_reduction_potential_V_vs_SHE": "Standard reduction potential (V vs SHE)",
    "Atomic_radius_pm": "Atomic radius (pm)",
    "Melting_point_C": "Melting point (°C)",
    "Boiling_point_C": "Boiling point (°C)",
    "Atomic_weight": "Atomic weight",
    "Group_number": "Group number",
    "Period": "Period",
}


def elements_in(text, known):
    """Element symbols from a free-text additive field, e.g. 'V; Cr; Hf' or 'Ga-In; Cu substrate'."""
    if not isinstance(text, str):
        return set()
    return {tok for tok in re.split(r"[^A-Za-z]+", text) if tok in known}


def element_evidence(analyzed, desc, target_class):
    """Per element: literature status and the product classes reported with it."""
    known = set(desc["Element"].astype(str))
    reported = {el: set() for el in known}
    for _, rec in analyzed.iterrows():
        for el in elements_in(rec.get("Additive_metal_clean"), known):
            reported[el].add(str(rec.get("Product_class", "Not reported")))

    out = []
    for el in desc["Element"].astype(str):
        classes = reported.get(el, set())
        if el in BASE_LM_ELEMENTS:
            status = STATUS_BASE
        elif target_class in classes:
            status = STATUS_TARGET
        elif classes:
            status = STATUS_OTHER
        else:
            status = STATUS_UNTESTED
        out.append({"Element": el, "Status": status, "Reported_products": ", ".join(sorted(classes)) or "-"})
    return pd.DataFrame(out)


def similarity_ranking(desc, evidence, features):
    """Rank untested elements by distance to the nearest target-product element (z-scored features)."""
    df = desc[["Element"] + features].merge(evidence, on="Element")
    complete = df.dropna(subset=features).copy()
    excluded = sorted(set(df["Element"]) - set(complete["Element"]))
    if complete.empty:
        return pd.DataFrame(), excluded

    z = complete[features].astype(float)
    z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
    complete[[f"z_{f}" for f in features]] = z.values

    refs = complete[complete["Status"] == STATUS_TARGET]
    cands = complete[complete["Status"] == STATUS_UNTESTED]
    if refs.empty or cands.empty:
        return pd.DataFrame(), excluded

    zcols = [f"z_{f}" for f in features]
    ref_mat = refs[zcols].to_numpy()
    rows = []
    for _, c in cands.iterrows():
        d = np.linalg.norm(ref_mat - c[zcols].to_numpy(dtype=float), axis=1)
        i = int(np.argmin(d))
        rows.append(
            {
                "Element": c["Element"],
                "Nearest target-product element": refs.iloc[i]["Element"],
                "Descriptor distance": float(d[i]),
                "Similarity score": float(1.0 / (1.0 + d[i])),
            }
        )
    ranking = pd.DataFrame(rows).sort_values("Descriptor distance").reset_index(drop=True)
    ranking.index = ranking.index + 1
    return ranking, excluded


def screening_map_data(desc, evidence, x, y):
    """One row per element with both descriptors present: Element, x_value, y_value, Status."""
    df = desc[["Element", x, y]].merge(evidence[["Element", "Status"]], on="Element").dropna(subset=[x, y])
    for c in (x, y):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=[x, y]).rename(columns={x: "x_value", y: "y_value"})
