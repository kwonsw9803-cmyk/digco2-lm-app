"""DigCO2-LM interface: navigation by evidence level, page headers and the reaction atlas.

Organising idea: every page belongs to one evidence level, so a reader always knows whether they are looking at
something published, computed, modelled, written by AI, or tested in a lab.
"""

from html import escape

import pandas as pd
import streamlit as st

LEVELS = {
    "Reported": "Published papers and reference tables",
    "Computed": "Public first-principles data",
    "Modelled": "Output of our own statistics and classifier",
    "AI-written": "Text written by a language model",
    "Tested": "Experiments submitted by users",
}

# (section, key used in app.py, title in the menu, url path)
NAV = [
    ("", "Overview", "Overview", "overview"),
    ("", "Evidence levels", "Evidence levels", "evidence"),
    ("Reported", "Database", "Records", "records"),
    ("Reported", "Literature Agent", "Ask the records", "ask"),
    ("Reported", "Descriptor Explorer", "Element properties", "elements"),
    ("Computed", "Computational Data", "Computed properties", "computed"),
    ("Modelled", "Descriptor Screening", "Element map", "element-map"),
    ("Modelled", "ML Prediction", "Product class", "product-class"),
    ("Modelled", "Recommendation Agent", "Candidates", "candidates"),
    ("AI-written", "Gemini Agent Report", "Stored reports", "stored-reports"),
    ("AI-written", "Live Claude Generator", "New report", "new-report"),
    ("Tested", "Experimental Feedback", "Submit a result", "submit"),
    ("About", "About", "About and references", "about"),
]
ADMIN_NAV = [("About", "File Status", "File status", "files")]

CSS = """
<style>
.lm-level {
    display: inline-block; margin-bottom: 0.25rem; padding: 0.1rem 0.5rem;
    font-family: "Source Code Pro", monospace; font-size: 0.8rem;
    border: 1px solid color-mix(in srgb, currentColor 35%, transparent);
    border-left: 4px solid #c8501f;
}
.lm-level span { opacity: 0.72; }
.lm-scroll { overflow-x: auto; }
table.lm-atlas { border-collapse: collapse; width: 100%; min-width: 50rem; table-layout: fixed; font-size: 0.9rem; }
table.lm-atlas thead th:first-child { width: 8.5rem; }
table.lm-atlas thead th:nth-child(2) { width: 24%; }
table.lm-atlas th, table.lm-atlas td {
    border: 0; border-bottom: 1px solid color-mix(in srgb, currentColor 18%, transparent);
    padding: 0.5rem 0.45rem; text-align: left; vertical-align: top;
}
table.lm-atlas thead th { vertical-align: bottom; border-bottom: 2px solid currentColor; font-weight: 600; line-height: 1.25; }
table.lm-atlas tbody th { font-weight: 600; }
table.lm-atlas td { border-left: 1px solid color-mix(in srgb, currentColor 10%, transparent); }
table.lm-atlas .n { display: block; font-weight: 400; font-size: 0.8rem; opacity: 0.72; }
.lm-recs { display: flex; flex-wrap: wrap; gap: 4px; align-items: flex-start; align-content: flex-start; }
.lm-rec, a.lm-rec:link, a.lm-rec:visited {
    display: inline-block; min-width: 2rem; padding: 0.15rem 0; text-align: center; text-decoration: none;
    font-family: "Source Code Pro", monospace; font-size: 0.85rem; color: inherit;
    background: color-mix(in srgb, currentColor 10%, transparent);
}
a.lm-rec:hover, a.lm-rec:focus-visible { background: #a8431a; color: #ffffff; outline: 2px solid #a8431a; outline-offset: 1px; }
@media (max-width: 640px) {
    table.lm-atlas { min-width: 0; display: block; }
    table.lm-atlas thead { position: absolute; left: -9999px; }
    table.lm-atlas tbody, table.lm-atlas tr { display: block; }
    table.lm-atlas tr { padding: 0.6rem 0; border-bottom: 1px solid color-mix(in srgb, currentColor 30%, transparent); }
    table.lm-atlas tbody th { display: block; border: 0; padding: 0 0 0.3rem 0; font-size: 1rem; }
    table.lm-atlas tbody th .n { display: inline; margin-left: 0.4rem; }
    table.lm-atlas td { display: flex; gap: 0.75rem; border: 0; padding: 0.2rem 0; }
    table.lm-atlas td::before { content: attr(data-label); flex: 0 0 8.5rem; }
    table.lm-atlas td.e { display: none; }
}
span.lm-rec { border: 1px dashed color-mix(in srgb, currentColor 45%, transparent); background: transparent; }
</style>
"""


def build_navigation(admin=False):
    """Create the menu. Returns (key of the current page, {key: st.Page})."""
    pages, sections, by_path = {}, {}, {}
    for section, key, title, path in NAV + (ADMIN_NAV if admin else []):
        # The page bodies live in app.py; the page function itself has nothing to do.
        p = st.Page(lambda: None, title=title, url_path=path, default=key == "Overview")
        pages[key] = p
        by_path[p.url_path] = key
        sections.setdefault(section, []).append(p)
    current = st.navigation(sections, expanded=True)
    return by_path.get(current.url_path, "Overview"), pages


def chart_ink():
    """Text colour for chart labels in the viewer's current theme."""
    return "#e7eaec" if st.context.theme.type == "dark" else "#16191d"


def page_header(title, level=None, lead=None):
    if level:
        st.markdown(
            f'<div class="lm-level"><b>{escape(level)}</b> <span>{escape(LEVELS[level])}</span></div>',
            unsafe_allow_html=True,
        )
    st.title(title, anchor=False)
    if lead:
        st.markdown(lead)


def reaction_atlas(analyzed, links=None):
    """HTML table: reaction mode x product class, one chip per record linking to its paper."""
    df = analyzed.assign(
        _row=analyzed["Reaction_type_clean"].fillna("Not reported").astype(str),
        _col=analyzed["Product_class"].fillna("Not reported").astype(str),
    )
    rows, cols = df["_row"].value_counts(), df["_col"].value_counts()
    row_order = sorted(rows.index, key=lambda r: (-rows[r], r))
    col_order = sorted(cols.index, key=lambda c: (-cols[c], c))
    links = links or {}

    def chip(r):
        rid = str(r["Record_ID"])
        year = "" if pd.isna(r.get("Year")) else f" ({str(r['Year']).split('.')[0]})"
        title = escape(f"{r.get('Paper_title', '')}{year}", quote=True)
        url = str(links.get(rid, ""))
        if url.startswith(("https://", "http://")):
            return (
                f'<a class="lm-rec" href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer" '
                f'title="{title}" aria-label="Record {escape(rid)}: {title}">{escape(rid)}</a>'
            )
        return f'<span class="lm-rec" title="{title} (no link recorded)">{escape(rid)}</span>'

    head = "".join(f'<th scope="col">{escape(c)}<span class="n">{cols[c]}</span></th>' for c in col_order)
    body = ""
    for r in row_order:
        cells = ""
        for c in col_order:
            hit = df[(df["_row"] == r) & (df["_col"] == c)]
            chips = "".join(chip(x) for _, x in hit.iterrows())
            label = escape(c, quote=True)
            cells += (
                f'<td data-label="{label}"><div class="lm-recs">{chips}</div></td>' if chips else '<td class="e"></td>'
            )
        body += f'<tr><th scope="row">{escape(r)}<span class="n">{rows[r]}</span></th>{cells}</tr>'
    return (
        '<div class="lm-scroll"><table class="lm-atlas"><thead><tr><th scope="col">Driven by</th>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def single_product_modes(analyzed, min_records=2):
    """[(reaction mode, n, product)] for modes whose records all report one product class."""
    out = []
    for mode, g in analyzed.groupby("Reaction_type_clean"):
        if len(g) >= min_records and g["Product_class"].nunique() == 1:
            out.append((str(mode), len(g), str(g["Product_class"].iloc[0])))
    return sorted(out, key=lambda t: -t[1])
