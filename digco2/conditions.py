"""Reaction conditions and product distributions of the literature records.

Two sheets of the Final workbook hold them: Conditions (one row per record and condition) and Products (one row per
condition and product). Nothing here fills a gap: a product without a number stays without a number, and a sum that
is not 100 is left as it is.
"""

import pandas as pd

FE = "Faradaic efficiency"
COUNTED = ("measured", "by difference")  # evidence that counts towards a sum
NOT_FROM_CO2 = ("H2", "O2")  # made from water, so never the main CO2 product
NOT_DETERMINED = "Not determined"
DERIVED = ["FE_sum_percent", "Main_CO2_product", "Solid_carbon"]


def _blank(v):
    return v is None or pd.isna(v) or str(v).strip() == ""


def value_text(r):
    """Readable value of one Products row, e.g. '93 ± 1.3 %', '>80 %', '40 to 80 %', or the kind of evidence."""
    if _blank(r["Value"]):
        return str(r["Evidence"])
    q = "" if _blank(r["Qualifier"]) else str(r["Qualifier"])
    unit = "" if _blank(r["Unit"]) else f" {r['Unit']}"
    text = f"{float(r['Value']):g}"
    if not _blank(r["Value_to"]):
        text += f" to {float(r['Value_to']):g}"
    if not _blank(r["Uncertainty"]):
        text += f" ± {float(r['Uncertainty']):g}"
    for word, sign in (("greater than", ">"), ("less than", "<"), ("approx.", "≈"), ("up to", "up to ")):
        if q.startswith(word):
            text, q = sign + text, q[len(word):].strip(" ,")
            break
    text += unit
    if q:
        text += f" ({q})"
    if r["Evidence"] not in ("measured",):
        text += f", {r['Evidence']}"
    return text


def balance(products):
    """Per condition: sum of the efficiencies that have a number, the main CO2 product, and what is said about solid carbon.

    A main product is named only when nothing unmeasured could be larger: its efficiency must exceed every other CO2
    product and the part of the charge that no number accounts for. That part is zero when the paper says the rest is
    H2, or that no other product was found.
    """
    out = []
    for cid, g in products.groupby("Condition_ID", sort=False):
        fe = g[g["Metric"] == FE]
        value = pd.to_numeric(fe["Value"], errors="coerce")
        single = value.notna() & fe["Value_to"].map(_blank) & fe["Evidence"].isin(COUNTED) & (fe["Qualifier"].fillna("") != "less than")
        counted = fe.assign(v=value)[single]
        total = counted["v"].sum() if len(counted) else None
        main = None
        if len(fe):
            main = NOT_DETERMINED
            co2 = counted[~counted["Product"].isin(NOT_FROM_CO2)].sort_values("v", ascending=False)
            if len(co2):
                closed = ((fe["Evidence"] == "stated as the remainder") & fe["Product"].isin(NOT_FROM_CO2)).any() or (
                    (fe["Evidence"] == "not detected") & (fe["Product"] == "other products")
                ).any()
                hidden = 0.0 if closed else max(0.0, 100.0 - total)
                rivals = [hidden] + co2["v"].iloc[1:].tolist()
                if co2["v"].iloc[0] > max(rivals):
                    top = co2.iloc[0]
                    main = top["Product"] + ("" if top["Evidence"] == "measured" else f" ({top['Evidence']})")
        carbon = g[g["Product"] == "solid carbon"]
        said = "; ".join(
            value_text(r) + ("" if _blank(r["Value"]) or r["Metric"] == FE else f" ({r['Metric']})") for _, r in carbon.iterrows()
        )
        out.append({"Condition_ID": cid, "FE_sum_percent": None if total is None else round(float(total), 2), "Main_CO2_product": main, "Solid_carbon": said or None})
    return pd.DataFrame(out, columns=["Condition_ID"] + DERIVED)


def _num(v):
    return "" if _blank(v) else f"{float(v):g}".replace("-", "\u2212")


def _span(a, b):
    a, b = _num(a), _num(b)
    return a if b in ("", a) else f"{a} to {b}"


def _join(*parts):
    return ", ".join(str(x) for x in parts if not _blank(x))


def conditions_view(conditions):
    """Readable table of Conditions rows. Columns that are empty for all of them are left out."""
    rows = []
    for _, r in conditions.iterrows():
        rows.append({
            "Condition": r["Condition"],
            "Main CO\u2082 product": r["Main_CO2_product"],
            "Sum of efficiencies (%)": _num(r["FE_sum_percent"]),
            "Solid carbon": r["Solid_carbon"],
            "CO\u2082 conversion (%)": _span(r["CO2_conversion_percent"], r["CO2_conversion_percent_to"]),
            "Rate": "" if _blank(r["Rate_value"]) else f"{_num(r['Rate_value'])} {r['Rate_unit']}",
            "Temperature (\u00b0C)": _span(r["Temp_C_min"], r["Temp_C_max"]),
            "Potential (V)": _join(_span(r["Potential_V"], r["Potential_V_to"]), r["Potential_role"]),
            "Reference electrode": r["Reference_electrode"],
            "Current density (mA/cm\u00b2)": _join(_num(r["Current_density_mA_cm2"]), r["Current_density_kind"]),
            "Time (h)": _num(r["Time_h"]),
            "CO\u2082 flow (mL/min)": _num(r["CO2_flow_mL_min"]),
            "CO\u2082 feed": r["CO2_feed"],
            "Medium": r["Medium"],
            "Pressure": r["Pressure"],
            "Page": r["Page"],
            "Still to check": r["Check_needed"],
            "Note": r["Note"],
        })
    out = pd.DataFrame(rows).map(lambda v: "" if _blank(v) else str(v))
    return out[[c for c in out.columns if (out[c] != "").any()]]


def products_view(products):
    """Readable table of Products rows."""
    out = pd.DataFrame({
        "Condition": products["Condition"],
        "Product": products["Product"],
        "Value": [value_text(r) for _, r in products.iterrows()],
        "Measure": products["Metric"],
        "Method": products["Method"],
        "Page": products["Page"],
        "Note": products["Note"],
    })
    return out.map(lambda v: "" if _blank(v) else str(v))
