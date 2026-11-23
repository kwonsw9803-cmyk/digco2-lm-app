"""Write the reviewed condition and product tables into the Final workbook and correct single cells of existing records.

Usage (from the app folder):
    python scripts/apply_conditions.py 2026-11-24

Input, all in data/review, named with that date:
  conditions_proposed_<date>.csv   one row per record and condition
  products_proposed_<date>.csv     one row per condition and product
  record_corrections_<date>.csv    cells of Core_database / Normalized_database to change, with the old value
  record_entries_<date>.csv        records whose product class depends on the condition: one entry (row) per condition

Updated:
  Final_LM_CO2_platform_database_with_descriptors.xlsx: Conditions and Products (replaced as a whole), the corrected
      cells and the entries of Normalized_database and Electrochemical_subset, Core_database, Data_quality_summary, README
  Step3_LM_CO2_database_analysis.xlsx: the same cells of Analyzed_database, and its derived sheets
Not updated: Step4_ML_ready_dataset.xlsx and the model. Product_class changes only where record_entries says so.

A cell is changed only if it still holds the old value of the corrections file; anything else stops the script before
it writes. Running it twice changes nothing.
"""

import sys
from copy import copy
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_literature_records as alr  # noqa: E402
from digco2 import conditions as cond_mod  # noqa: E402

FINAL, STEP3 = alr.FINAL, alr.STEP3
REVIEW = alr.APP / "data" / "review"
NUMBER_COLUMNS = {"Voltage_V", "Current_density_mA_cm2", "Reaction_time_h_value", "FE_HCOOH_%", "FE_CO_%", "FE_carbon_%", "FE_total_%"}
TEXT_COLUMNS = {"Condition_ID", "Condition", "Potential_role", "Reference_electrode", "Current_density_kind", "CO2_feed", "Medium", "Rate_unit", "Pressure",
                "Basis", "Page", "Check_needed", "Note", "Product", "Metric", "Unit", "Qualifier", "Evidence", "Method",
                "Main_CO2_product", "Solid_carbon"}
README = {
    "Conditions": "One row per record and condition, read from the papers (page given). Blank = not found in the main text; supporting "
                  "information was not read. Potential, reference electrode and current density apply to electrochemical rows only.",
    "Products": "One row per condition and product. Only numbers stated by the authors; Evidence says how each was obtained. A product that "
                "was not measured is not zero, and sums are not forced to 100 (a lower bound counts as written). Not used to train the classifier.",
}
WIDTH = {"Condition_ID": 12, "Record_ID": 10, "Condition": 34, "Medium": 30, "Note": 60, "Check_needed": 40, "Basis": 22, "Evidence": 24, "Method": 22,
         "Reference_electrode": 24, "CO2_feed": 26, "Main_CO2_product": 22, "Solid_carbon": 30, "Product": 18, "Metric": 20, "Pressure": 18, "Rate_unit": 24}


def read(name, date):
    return pd.read_csv(REVIEW / f"{name}_{date}.csv", dtype=str, encoding="utf-8-sig").fillna("")


def typed(df):
    """Numbers as numbers, blanks as None."""
    out = df.copy().astype(object)
    for c in out.columns:
        for i in out.index:
            v = out.at[i, c]
            if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
                out.at[i, c] = None
            elif c not in TEXT_COLUMNS and not isinstance(v, (int, float)):
                f = float(v)  # a non-number in a number column is an error
                out.at[i, c] = int(f) if f == int(f) else f
    return out


def same(cell, text):
    """Does this cell hold the value written as text in the corrections file?"""
    if cell is None or str(cell) == "":
        return text == ""
    if alr.is_num(cell) and alr.is_num(text):
        return float(cell) == float(text)
    return str(cell) == text


def correct(ws, record_id, column, old, new, mode, numeric):
    """Change one cell. Returns 'changed', 'already' or a message about what stands in the way."""
    head = alr.header(ws)
    rows = [r for r in range(2, alr.last_data_row(ws) + 1) if str(ws.cell(r, head.index("Record_ID") + 1).value) == str(record_id)]
    if len(rows) != 1:
        return f"{ws.title}: record {record_id} found {len(rows)} times"
    cell = ws.cell(rows[0], head.index(column) + 1)
    if mode == "replace sentence":
        text = "" if cell.value is None else str(cell.value)
        if new in text:
            return "already"
        if text.count(old) != 1:
            return f"{ws.title} {record_id} {column}: sentence not found in {text!r}"
        cell.value = text.replace(old, new)
        return "changed"
    if same(cell.value, new):
        return "already"
    if not same(cell.value, old):
        return f"{ws.title} {record_id} {column}: holds {cell.value!r}, expected {old!r}"
    if numeric and alr.is_num(new):
        f = float(new)
        cell.value = int(f) if f == int(f) else f
    else:
        cell.value = new
    return "changed"


def write_sheet(wb, title, df, like, after):
    """Replace a sheet as a whole, in the header style of an existing one."""
    if title in wb.sheetnames:
        pos = wb.sheetnames.index(title)
        del wb[title]
    else:
        pos = wb.sheetnames.index(after) + 1
    ws = wb.create_sheet(title, pos)
    model = wb[like]
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))
    for i, name in enumerate(df.columns, start=1):
        src, dst = model.cell(1, 1), ws.cell(1, i)
        dst.font, dst.fill, dst.border = copy(src.font), copy(src.fill), copy(src.border)
        dst.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(i)].width = WIDTH.get(name, 14)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "D2"
    table = Table(displayName=f"{title}Table", ref=f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}")
    style = next(iter(model.tables.values())).tableStyleInfo
    table.tableStyleInfo = TableStyleInfo(name=style.name if style else "TableStyleMedium2", showRowStripes=True)
    ws.add_table(table)


def set_entries(ws, entries):
    """Give every row an Entry label and the listed records one row per entry (same values except those listed)."""
    head = alr.header(ws)
    if "Entry" not in head:
        ws.insert_cols(2)
        widths = {i: ws.column_dimensions[get_column_letter(i)].width for i in range(2, len(head) + 1)}
        for i, w in widths.items():
            ws.column_dimensions[get_column_letter(i + 1)].width = w
        ws.column_dimensions["B"].width = 20
        for r in range(1, alr.last_data_row(ws) + 1):
            src, dst = ws.cell(r, 1), ws.cell(r, 2)
            dst.value = "Entry" if r == 1 else str(src.value)
            if src.has_style:
                dst._style = copy(src._style)
        head = alr.header(ws)
    col = {h: i for i, h in enumerate(head, start=1)}
    for rid, variants in entries.items():
        rows = [r for r in range(2, alr.last_data_row(ws) + 1) if str(ws.cell(r, col["Record_ID"]).value) == rid]
        if not rows:
            continue
        for _ in range(len(variants) - len(rows)):
            new = rows[-1] + 1
            ws.insert_rows(new)
            for c in range(1, len(head) + 1):
                src, dst = ws.cell(rows[0], c), ws.cell(new, c)
                dst.value = src.value
                if src.has_style:
                    dst._style = copy(src._style)
            rows.append(new)
        for r, (label, values) in zip(rows, variants.items()):
            ws.cell(r, col["Entry"], label)
            for c, v in values.items():
                number = float(v) if c in NUMBER_COLUMNS and alr.is_num(v) else None
                ws.cell(r, col[c], v if number is None else int(number) if number == int(number) else number)


def fit_tables(ws):
    """Make each Excel table cover the whole header and all data rows."""
    for t in ws.tables.values():
        ref = f"A1:{get_column_letter(len([h for h in alr.header(ws) if h is not None]))}{alr.last_data_row(ws)}"
        if t.ref != ref:
            t.ref, t.autoFilter, t.tableColumns = ref, None, []


def frame(ws):
    head = alr.header(ws)
    return pd.DataFrame([[c.value for c in row] for row in ws.iter_rows(min_row=2, max_row=alr.last_data_row(ws))], columns=head)


def current_state():
    step3 = pd.read_excel(STEP3, sheet_name=None)
    labels = pd.read_excel(FINAL, sheet_name="Data_quality_summary", header=None)
    vals = {"core": pd.read_excel(FINAL, sheet_name="Core_database"), "norm": pd.read_excel(FINAL, sheet_name="Normalized_database"),
            "labels": dict(zip(labels[0], labels[1]))}
    return step3, vals


def main(date):
    step3, vals = current_state()
    alr.check_reproduces(step3, step3["Analyzed_database"], vals)
    conditions, products, corrections = read("conditions_proposed", date), read("products_proposed", date), read("record_corrections", date)
    entries = {}
    for _, e in read("record_entries", date).iterrows():
        entries.setdefault(e["Record_ID"], {}).setdefault(e["Entry"], {})[e["Column"]] = e["Value"]
    known = set(vals["core"]["Record_ID"].dropna().astype(int).astype(str))
    assert set(conditions["Record_ID"]) == known, "every record needs at least one condition"
    assert not conditions["Condition_ID"].duplicated().any() and set(products["Condition_ID"]) <= set(conditions["Condition_ID"])
    assert (products["Page"] != "").all(), "every product row needs a page"

    products = typed(products)
    conditions = conditions.drop(columns="Change").merge(cond_mod.balance(products), on="Condition_ID", how="left")
    front = [c for c in conditions.columns if c not in ("Basis", "Page", "Check_needed", "Note")]
    conditions = typed(conditions[front + ["Basis", "Page", "Check_needed", "Note"]])

    wb, wb3 = openpyxl.load_workbook(FINAL), openpyxl.load_workbook(STEP3)
    log, problems = {"changed": 0, "already": 0}, []
    for _, k in corrections.iterrows():
        if k["Table"] == "Core":
            targets = [wb["Core_database"]]
        else:
            targets = [wb["Normalized_database"], wb3["Analyzed_database"]]
            if k["Record_ID"] in set(frame(wb["Electrochemical_subset"])["Record_ID"].astype(str)):
                targets.append(wb["Electrochemical_subset"])
        for ws in targets:
            res = correct(ws, k["Record_ID"], k["Column"], k["Old"], k["New"], k["Mode"], k["Table"] != "Core")
            if res in log:
                log[res] += 1
            else:
                problems.append(res)
    if problems:
        raise SystemExit("Nothing written:\n  " + "\n  ".join(problems))

    for ws in (wb["Normalized_database"], wb["Electrochemical_subset"], wb3["Analyzed_database"]):
        set_entries(ws, entries)
    for name in ("Core_database", "Mechanism_notes", "Normalized_database", "Electrochemical_subset"):
        fit_tables(wb[name])
    write_sheet(wb, "Conditions", conditions, like="Normalized_database", after="Electrochemical_subset")
    write_sheet(wb, "Products", products, like="Normalized_database", after="Conditions")
    ws = wb["README"]
    for label, text in README.items():
        row = next((r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == label), None)
        if row is None:
            row = alr.last_data_row(ws) + 1
            for col in (1, 2):
                src, dst = ws.cell(row - 1, col), ws.cell(row, col)
                dst.font, dst.fill, dst.border, dst.alignment = copy(src.font), copy(src.fill), copy(src.border), copy(src.alignment)
        ws.cell(row, 1, label)
        ws.cell(row, 2, text)

    norm, core, analyzed = frame(wb["Normalized_database"]), frame(wb["Core_database"]), frame(wb3["Analyzed_database"])
    fc = alr.final_counts(core, norm)
    for label, v in fc["quality"].items():
        alr.set_label_value(wb["Data_quality_summary"], 1, 2, label, v)
    alr.write_count_block(wb["Data_quality_summary"], 1, 2, "Reaction_type_clean", fc["quality_reaction"])
    alr.write_count_block(wb["Data_quality_summary"], 1, 2, "Product_class", fc["quality_product"])
    split = "; ".join(f"record {r} has {len(v)} entries" for r, v in entries.items())
    row = alr.set_label_value(wb["Data_quality_summary"], 1, 2, "Total records", fc["quality"]["Total records"])
    wb["Data_quality_summary"].cell(row, 3, f"Records in Normalized_database ({split}; Product_class counts entries)")
    derived, summary = alr.step3_derived(analyzed)
    ws = wb3["Analysis_summary"]
    for label, v in summary:
        if not any(ws.cell(r, 1).value == label for r in range(1, ws.max_row + 1)):
            ws.append([label, v, f"{split}; product tables count entries"])
    for name, df in derived.items():
        alr.rewrite(wb3[name], df)
    for label, v in summary:
        alr.set_label_value(wb3["Analysis_summary"], 1, 2, label, v)
    if wb.calculation is not None:
        wb.calculation.fullCalcOnLoad = True
    wb.save(FINAL)
    wb3.save(STEP3)

    step3, vals = current_state()
    alr.check_reproduces(step3, step3["Analyzed_database"], vals)
    print(f"entries: {split}")
    print(f"saved: {len(conditions)} conditions, {len(products)} product rows; cells changed {log['changed']}, already correct {log['already']}; derived sheets reproduce")


if __name__ == "__main__":
    main(sys.argv[1])
