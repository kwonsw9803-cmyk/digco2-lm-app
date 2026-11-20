"""Rebuild data/computational_data.csv from OQMD convex hulls (X-O, X-C, Ga-X) for the descriptor elements.

Usage: python scripts/fetch_oqmd.py data/computational_data.csv oqmd_raw.json Final_LM_CO2_platform_database_with_descriptors.xlsx
Responses are cached in an oqmd_cache/ folder next to the raw JSON, so an interrupted run can resume.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

ELEMENTS = ["Ga", "In", "Sn", "Mg", "Li", "Na", "Ce", "V", "Cu", "Ag", "Te", "Bi", "Fe", "Mn", "Ca", "Ni", "Zn", "Al"]
OUT = sys.argv[1]
RAW = sys.argv[2]
DESC = pd.read_excel(sys.argv[3], sheet_name="Element_descriptors").set_index("Element")["Oxide_formula"].to_dict()
API = "https://oqmd.org/oqmdapi/formationenergy"


import os

CACHE = os.path.join(os.path.dirname(RAW), "oqmd_cache")
os.makedirs(CACHE, exist_ok=True)


def hull(system):
    path = os.path.join(CACHE, f"{system}.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d["data"], d["meta"]["time_stamp"]
    q = urllib.parse.urlencode(
        {"composition": system, "fields": "name,entry_id,delta_e,stability,natoms", "filter": "stability=0", "limit": 200}
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=180) as r:
                d = json.load(r)
            if d.get("response_message") != "OK" or d["meta"].get("more_data_available"):
                raise RuntimeError(f"incomplete response for {system}")
            with open(path, "w") as f:
                json.dump(d, f)
            return d["data"], d["meta"]["time_stamp"]
        except Exception as exc:  # retry on transient errors
            print("retry", system, exc, file=sys.stderr, flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(system)


def parse(name):
    return {el: float(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", name)}


raw, rows = {}, []
stamp = None


def add(prop, unit, solute, system, value, phase, entry, note):
    rows.append(
        {
            "Record_ID": f"P-{len(rows) + 1:04d}",
            "System": system,
            "Base_LM": "Ga" if prop == "Ga_X_compound_formation_energy" else "",
            "Solute": solute,
            "Solute_at_pct": None,
            "Temperature_K": 0,
            "Method": "DFT, high-throughput (OQMD convex hull)",
            "Functional_or_potential": "GGA-PBE (+U for selected oxides), OQMD reference energies",
            "Property": prop,
            "Value": round(value, 4),
            "Unit": unit,
            "Notes": note,
            "Source": f"OQMD entry {entry} ({phase}): https://oqmd.org/materials/entry/{entry}" if entry else "OQMD convex hull query (no stable compound)",
        }
    )


for el in ELEMENTS:
    # oxides: formation energy per O atom of the most O-stabilising hull oxide
    data, stamp = hull(f"{el}-O")
    raw[f"{el}-O"] = data
    ox = []
    for p in data:
        comp = parse(p["name"])
        if set(comp) == {el, "O"}:
            n_atoms = sum(comp.values())
            ox.append((p["delta_e"] * n_atoms / comp["O"], p["name"], p["entry_id"], p["delta_e"]))
    listed = {f.strip() for f in str(DESC.get(el, "")).split(",")}
    known = [o for o in ox if o[1] in listed]
    if known:
        v, name, eid, de = min(known)
        add("Oxide_formation_energy_per_O", "eV/O atom", el, f"{el}–O (bulk)", v, name, eid,
            f"{name} (oxide listed in the descriptor table, on the OQMD hull): ΔH_f = {de:.3f} eV/atom; Value is per O atom")
    elif ox:
        v, name, eid, de = min(ox)
        add("Oxide_formation_energy_per_O", "eV/O atom", el, f"{el}–O (bulk)", v, name, eid,
            f"No listed oxide on the hull; most negative ΔH_f per O among {len(ox)} hull oxide(s): {name} ({de:.3f} eV/atom)")

    # carbides: most negative formation energy per atom on the X-C hull (0 = no stable carbide)
    data, stamp = hull(f"{el}-C")
    raw[f"{el}-C"] = data
    cb = [(p["delta_e"], p["name"], p["entry_id"]) for p in data if set(parse(p["name"])) == {el, "C"}]
    if cb:
        v, name, eid = min(cb)
        add("Carbide_formation_energy", "eV/atom", el, f"{el}–C (bulk)", v, name, eid,
            f"Most negative ΔH_f on the {el}–C hull ({len(cb)} stable carbide(s))")
    else:
        add("Carbide_formation_energy", "eV/atom", el, f"{el}–C (bulk)", 0.0, None, None,
            f"No stable binary carbide on the {el}–C hull; value set to 0")

    # Ga-X intermetallics: most negative formation energy per atom on the Ga-X hull
    if el != "Ga":
        data, stamp = hull(f"Ga-{el}")
        raw[f"Ga-{el}"] = data
        gx = [(p["delta_e"], p["name"], p["entry_id"]) for p in data if set(parse(p["name"])) == {"Ga", el}]
        if gx:
            v, name, eid = min(gx)
            add("Ga_X_compound_formation_energy", "eV/atom", el, f"Ga–{el} (bulk)", v, name, eid,
                f"Most negative ΔH_f on the Ga–{el} hull ({len(gx)} stable compound(s)); proxy for solute–Ga interaction")
        else:
            add("Ga_X_compound_formation_energy", "eV/atom", el, f"Ga–{el} (bulk)", 0.0, None, None,
                f"No stable Ga–{el} compound on the hull; value set to 0 (weak solute–Ga interaction)")
    print(el, "done", file=sys.stderr)

df = pd.DataFrame(rows)
df["Notes"] = df["Notes"] + f" · retrieved {stamp[:10]} via OQMD API (CC BY 4.0)"
df.to_csv(OUT, index=False, encoding="utf-8")
with open(RAW, "w") as f:
    json.dump({"time_stamp": stamp, "hulls": raw}, f, indent=1)
print(df[["Solute", "Property", "Value", "Source"]].to_string())
