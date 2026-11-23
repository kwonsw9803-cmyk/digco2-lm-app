"""Add curated literature records to the platform workbooks and refresh the derived sheets.

Usage (from the app folder):
    python scripts/add_literature_records.py data/review/literature_additions_2026-11-20.json

Input is a JSON file with "records" (new rows) and "doi_updates" (DOI fixes for existing rows); see
data/review/literature_additions_2026-11-20.json. Anything a record does not state is written as
"Not reported" (or "Not applicable" where the field cannot apply), matching the existing database policy.

Updated:
  Final_LM_CO2_platform_database_with_descriptors.xlsx: Core_database, Mechanism_notes, Normalized_database,
      Electrochemical_subset, Summary, Data_quality_summary
  Step3_LM_CO2_database_analysis.xlsx: Analyzed_database and every count / cross-tab sheet
Not updated: Step4_ML_ready_dataset.xlsx and the model (retraining is a separate, reviewed step).

Before writing, the script recomputes the derived sheets from the current rows and stops if they do not match
the stored sheets, so the refresh cannot silently change their definitions. Records whose DOI is already in
Core_database are skipped, so running the script twice adds nothing.
"""

import json
import sys
from pathlib import Path

import openpyxl
import pandas as pd

APP = Path(__file__).resolve().parents[1]
FINAL = APP / "Final_LM_CO2_platform_database_with_descriptors.xlsx"
STEP3 = APP / "Step3_LM_CO2_database_analysis.xlsx"
NR, NA = "Not reported", "Not applicable"
CARBON_COLS = ["Carbon_yield_value", "Carbon_rate_umol_h", "Carbon_content_%", "Weight_gain_mg"]
FE_COLS = ["FE_HCOOH_%", "FE_CO_%", "FE_carbon_%", "FE_total_%"]
EC_STEP3_COLS = ["Paper_title", "Year", "Additive_metal_clean", "Voltage_V", "Current_density_mA_cm2",
                 "FE_HCOOH_%", "FE_CO_%", "FE_carbon_%", "FE_total_%", "Product_class"]


def is_num(v):
    return pd.notna(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])


def temp_class(v):
    if not is_num(v):
        return NR
    t = float(v)
    if t < 100:
        return "low temperature (<100 °C)"
    if t <= 300:
        return "mild temperature (100–300 °C)"
    if t < 600:
        return "intermediate temperature (300–600 °C)"
    return "high temperature (≥600 °C)"


# ---------------------------------------------------------------- derived sheets
def crosstab(df, row):
    return pd.crosstab(df[row], df["Product_class"]).reset_index().rename_axis(None, axis=1)


def step3_derived(a):
    # A record may have several entries (rows) with their own product class. Product tables count entries, the rest records.
    one = a.drop_duplicates("Record_ID")
    counts = lambda df, col: df[col].value_counts().rename_axis(col).reset_index(name="Count")  # noqa: E731
    co2 = one["CO2_conversion_representative_%"].map(is_num)
    carbon = one[CARBON_COLS].apply(lambda r: any(str(v) not in (NR, NA) for v in r), axis=1)
    ec = a[a["Reaction_type_clean"] == "electrochemical"]
    summary = [
        ("Total records", len(one)),
        ("Number of product classes", a["Product_class"].nunique()),
        ("Most frequent product class", a["Product_class"].value_counts().index[0]),
        ("Number of reaction types", a["Reaction_type_clean"].nunique()),
        ("Electrochemical records", ec["Record_ID"].nunique()),
        ("Records with CO2 conversion data", int(co2.sum())),
        ("Records with carbon yield/rate data", int(carbon.sum())),
    ]
    if len(a) > len(one):
        summary.append(("Total entries", len(a)))
    avail = lambda s, name: s.value_counts().rename_axis(name).reset_index(name="Count").sort_values(name)  # noqa: E731
    return {
        "Product_class_count": counts(a, "Product_class"),
        "Reaction_type_count": counts(one, "Reaction_type_clean"),
        "Reaction_vs_Product": crosstab(a, "Reaction_type_clean"),
        "Additive_vs_Product": crosstab(a, "Additive_metal_clean"),
        "Temp_vs_Product": crosstab(a, "Temp_class"),
        "Electrochemical_subset": ec[EC_STEP3_COLS],
        "CO2_data_availability": avail(co2, "Has_CO2_conversion_data"),
        "Carbon_data_availability": avail(carbon, "Has_carbon_yield_or_rate_data"),
    }, summary


def final_counts(core, norm):
    fe_num = norm[FE_COLS].apply(lambda r: any(is_num(v) for v in r), axis=1)
    one = norm.drop_duplicates("Record_ID")
    return {
        "summary": {
            "Core experimental records": len(core),
            "Records with CO2_conversion_% reported": int((core["CO2_conversion_%"].astype(str) != NR).sum()),
            "Records with FE_% reported": int((core["FE_%"].astype(str) != NR).sum()),
            "Records with Temp_C not reported": int((core["Temp_C"].astype(str) == NR).sum()),
        },
        "summary_reactions": core["Reaction_type"].value_counts(),
        "quality": {
            "Total records": len(one),
            "CO2 conversion available": int(one["CO2_conversion_representative_%"].map(is_num).sum()),
            "Carbon yield/rate available": int(one[CARBON_COLS].apply(
                lambda r: any(str(v) not in (NR, NA) for v in r), axis=1).sum()),
            "Exact FE data available": int(norm.loc[fe_num, "Record_ID"].nunique()),
            "Normalized representative temperature available": int(one["Temp_representative_C"].map(is_num).sum()),
            "Electrochemical subset records": int((one["Reaction_type_clean"] == "electrochemical").sum()),
        },
        "quality_reaction": one["Reaction_type_clean"].value_counts().sort_index(),
        "quality_product": norm["Product_class"].value_counts().sort_index(),
    }


def same_table(stored, computed, key):
    """Compare as sets of rows keyed by the first column (row order of ties is arbitrary)."""
    s = stored.astype(str).set_index(key).sort_index()
    c = computed.astype(str).set_index(key).sort_index()
    return s.columns.tolist() == c.columns.tolist() and s.equals(c)


def check_reproduces(step3_sheets, a, wb_final_vals):
    derived, summary = step3_derived(a)
    for name, df in derived.items():
        stored = step3_sheets[name]
        key = stored.columns[0]
        if name == "Electrochemical_subset":
            ok = stored.astype(str).reset_index(drop=True).equals(df.astype(str).reset_index(drop=True))
        else:
            ok = same_table(stored, df, key)
        if not ok:
            raise SystemExit(f"Recomputed {name} differs from the stored sheet; not writing anything.")
    stored_summary = dict(zip(step3_sheets["Analysis_summary"]["Item"], step3_sheets["Analysis_summary"]["Value"]))
    for k, v in summary:
        if str(stored_summary.get(k)) != str(v):
            raise SystemExit(f"Analysis_summary '{k}' would change from {stored_summary.get(k)} to {v}; stopping.")
    fc = final_counts(wb_final_vals["core"], wb_final_vals["norm"])
    for k, v in fc["quality"].items():
        if str(wb_final_vals["labels"].get(k)) != str(v):
            raise SystemExit(f"'{k}' would change from {wb_final_vals['labels'].get(k)} to {v}; stopping.")


# ---------------------------------------------------------------- building rows
def build_rows(rec, rid, norm_cols, core_cols, mech_cols):
    ec = rec["Reaction_type_clean"] == "electrochemical"
    norm = {c: NR for c in norm_cols}
    norm.update({"Record_ID": rid, "Paper_type_clean": "Experimental", "Data_notes": None})
    if "Entry" in norm_cols:
        norm["Entry"] = str(rid)
    for c in ["Voltage_V", "Current_density_mA_cm2", "Reference_electrode"] + FE_COLS:
        norm[c] = NR if ec else NA
    # Carbon columns stay "Not reported" even for non-carbon products, as in the existing rows.
    for c in norm_cols:
        if c in rec:
            norm[c] = rec[c]
    if not is_num(norm["Temp_representative_C"]):
        norm["Temp_min_C"] = norm["Temp_max_C"] = NR
    source = f"Abstract via OpenAlex (retrieved 2026-11-20); doi:{rec['doi']}"
    norm["Data_notes"] = " ".join(x for x in [rec.get("Data_notes"), "Extracted from the abstract only; full text not checked."] if x)

    core = {c: NR for c in core_cols}
    core.update({
        "Record_ID": rid, "Paper_type": "Experimental", "Pressure": NR,
        "Voltage_or_current": NA if not ec else NR, "FE_%": NA if not ec else NR,
        "Carbon_yield_or_rate": NR,
        "Data_status": "Abstract only / partially extracted", "Source_file": source,
        "DOI_or_URL": f"https://doi.org/{rec['doi']}",
    })
    for c in core_cols:
        if c in rec:
            core[c] = rec[c]
    for c in ["Paper_title", "Year", "LM_composition", "Reactor_or_cell", "CO2_feed", "Main_product",
              "Characterization_methods", "Mechanism_keyword"]:
        if c in norm and c in core_cols:
            core[c] = norm[c]

    mech = {
        "Record_ID": rid, "Paper_title": rec["Paper_title"], "Active_species": rec.get("Active_species", NR),
        "Proposed_mechanism": rec["Mechanism_keyword"],
        "Evidence_methods": rec.get("Characterization_methods", NR),
        "Limitation_keyword": "Abstract-level extraction; check full text",
    }
    assert list(mech) == mech_cols, mech_cols
    analyzed = dict(norm, Temp_class=temp_class(norm["Temp_representative_C"]))
    return norm, core, mech, analyzed


# ---------------------------------------------------------------- writing helpers
def header(ws):
    return [c.value for c in ws[1]]


def last_data_row(ws):
    """Last row with any value; ws.max_row also counts formatted empty rows (Core_database is styled to row 100)."""
    return max((c.row for row in ws.iter_rows() for c in row if c.value not in (None, "")), default=0)


def append(ws, row_dict):
    r = last_data_row(ws) + 1
    for i, h in enumerate(header(ws), start=1):
        ws.cell(row=r, column=i, value=row_dict.get(h))


def rewrite(ws, df):
    """Keep the styled header row, replace everything below it."""
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for i, col in enumerate(df.columns, start=1):
        ws.cell(row=1, column=i, value=col)
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and pd.isna(v)) else (v.item() if hasattr(v, "item") else v)
                   for v in row])


def set_label_value(ws, label_col, value_col, label, value, start_row=1):
    for r in range(start_row, ws.max_row + 1):
        if ws.cell(r, label_col).value == label:
            ws.cell(r, value_col, value)
            return r
    raise SystemExit(f"Label '{label}' not found in {ws.title}")


def write_count_block(ws, label_col, value_col, title, series):
    """Rewrite a 'title / Count' block in place; the block must keep its length (asserted)."""
    for r in range(1, ws.max_row + 1):
        if ws.cell(r, label_col).value == title:
            start = r + 1
            break
    else:
        raise SystemExit(f"Block '{title}' not found in {ws.title}")
    end = start
    while end <= ws.max_row and ws.cell(end, label_col).value not in (None, ""):
        end += 1
    rows = list(series.items())
    if len(rows) > end - start:
        ws.insert_rows(end, len(rows) - (end - start))
    for i in range(max(len(rows), end - start)):
        k, v = rows[i] if i < len(rows) else (None, None)
        ws.cell(start + i, label_col, k)
        ws.cell(start + i, value_col, None if v is None else int(v))


def update_summary_formulas(wb, reaction_counts):
    """The Summary sheet is formula-driven (COUNTA/COUNTIF over Core_database). Keep it that way: extend the
    ranges to the last data row and list every Reaction_type, each with its own COUNTIF."""
    ws, core_ws = wb["Summary"], wb["Core_database"]
    last = last_data_row(core_ws)
    ref_last = last_data_row(wb["Reference_papers"])
    col = {h: openpyxl.utils.get_column_letter(i) for i, h in enumerate(header(core_ws), start=1)}
    rng = lambda name: f"Core_database!${col[name]}$2:${col[name]}${last}"  # noqa: E731
    formulas = {
        "Core experimental records": f"=COUNTA({rng('Record_ID')})",
        "Reference/background records": f"=COUNTA(Reference_papers!$A$2:$A${ref_last})",
        "Records with CO2_conversion_% reported": f'=COUNTIF({rng("CO2_conversion_%")},"<>Not reported")',
        "Records with FE_% reported": f'=COUNTIF({rng("FE_%")},"<>Not reported")',
        "Records with Temp_C not reported": f'=COUNTIF({rng("Temp_C")},"Not reported")',
    }
    for label, f in formulas.items():
        set_label_value(ws, 1, 2, label, f)
    write_count_block(ws, 4, 5, "Reaction_type", reaction_counts)
    start = next(r for r in range(1, ws.max_row + 1) if ws.cell(r, 4).value == "Reaction_type") + 1
    for r in range(start, start + len(reaction_counts)):
        ws.cell(r, 5, f"=COUNTIF({rng('Reaction_type')},D{r})")
    # openpyxl stores no cached results; ask Excel to recalculate everything on open.
    if wb.calculation is None:
        from openpyxl.workbook.properties import CalcProperties
        wb.calculation = CalcProperties()
    wb.calculation.fullCalcOnLoad = True


# ---------------------------------------------------------------- main
def main(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    core = pd.read_excel(FINAL, sheet_name="Core_database")
    norm = pd.read_excel(FINAL, sheet_name="Normalized_database")
    mech_df = pd.read_excel(FINAL, sheet_name="Mechanism_notes")
    step3 = pd.read_excel(STEP3, sheet_name=None)
    a = step3["Analyzed_database"]

    # Summary is formula-driven (its cached values vanish after an openpyxl save), so only check the value sheet.
    quality_ws = pd.read_excel(FINAL, sheet_name="Data_quality_summary", header=None)
    labels = dict(zip(quality_ws[0], quality_ws[1]))
    check_reproduces(step3, a, {"core": core, "norm": norm, "labels": labels})
    print("check: derived sheets reproduce from the current rows")

    known = set(core["DOI_or_URL"].astype(str).str.lower())
    next_id = int(core["Record_ID"].max()) + 1
    new_norm, new_core, new_mech, new_an = [], [], [], []
    for rec in spec["records"]:
        if f"https://doi.org/{rec['doi']}".lower() in known:
            print(f"skip {rec['doi']} (already in Core_database)")
            continue
        n, c, m, an = build_rows(rec, next_id, list(norm.columns), list(core.columns), list(mech_df.columns))
        new_norm.append(n); new_core.append(c); new_mech.append(m); new_an.append(an)
        print(f"add Record_ID {next_id}: {rec['Paper_title'][:70]}")
        next_id += 1

    core2 = pd.concat([core, pd.DataFrame(new_core)], ignore_index=True)
    for u in spec.get("doi_updates", []):
        idx = core2.index[core2["Record_ID"] == u["Record_ID"]]
        if len(idx) != 1:
            raise SystemExit(f"Record_ID {u['Record_ID']} not found")
        old = str(core2.at[idx[0], "DOI_or_URL"])
        if old != u["DOI_or_URL"]:
            print(f"DOI Record_ID {u['Record_ID']}: {old} -> {u['DOI_or_URL']}")
            core2.at[idx[0], "DOI_or_URL"] = u["DOI_or_URL"]
    norm2 = pd.concat([norm, pd.DataFrame(new_norm)], ignore_index=True)
    a2 = pd.concat([a, pd.DataFrame(new_an)], ignore_index=True)

    # ---- Final workbook
    wb = openpyxl.load_workbook(FINAL)
    for n, c, m in zip(new_norm, new_core, new_mech):
        append(wb["Core_database"], c)
        append(wb["Normalized_database"], n)
        append(wb["Mechanism_notes"], m)
        if n["Reaction_type_clean"] == "electrochemical":
            append(wb["Electrochemical_subset"], n)
    ws = wb["Core_database"]
    doi_col = header(ws).index("DOI_or_URL") + 1
    for u in spec.get("doi_updates", []):
        set_label_value(ws, 1, doi_col, u["Record_ID"], u["DOI_or_URL"], start_row=2)
    fc = final_counts(core2, norm2)
    update_summary_formulas(wb, fc["summary_reactions"])
    for k, v in fc["quality"].items():
        set_label_value(wb["Data_quality_summary"], 1, 2, k, v)
    write_count_block(wb["Data_quality_summary"], 1, 2, "Reaction_type_clean", fc["quality_reaction"])
    write_count_block(wb["Data_quality_summary"], 1, 2, "Product_class", fc["quality_product"])
    wb.save(FINAL)

    # ---- Step3 workbook
    wb3 = openpyxl.load_workbook(STEP3)
    for an in new_an:
        append(wb3["Analyzed_database"], an)
    derived, summary = step3_derived(a2)
    for name, df in derived.items():
        rewrite(wb3[name], df)
    for k, v in summary:
        set_label_value(wb3["Analysis_summary"], 1, 2, k, v)
    wb3.save(STEP3)
    print(f"saved: {len(core2)} core records ({len(new_core)} new), {len(spec.get('doi_updates', []))} DOI updates")


if __name__ == "__main__":
    main(sys.argv[1])
