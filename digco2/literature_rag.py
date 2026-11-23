"""Retrieval-augmented literature agent over the curated LM-CO2 records.

Retrieval is TF-IDF (word + character n-grams so formulas like 'GaInSn' or
'Ga2O3' match) over one document per record built from Core_database and
Mechanism_notes. Evidence is always shown; when Claude is connected the
LLM writes an answer that may only use the retrieved records and must cite them
as [R<id>].
"""

import numpy as np
import pandas as pd

DOC_FIELDS_CORE = [
    "Paper_title",
    "Year",
    "LM_composition",
    "Additive_active_component",
    "Reaction_type",
    "Reactor_or_cell",
    "Temp_C",
    "Pressure",
    "Voltage_or_current",
    "Main_product",
    "CO2_conversion_%",
    "FE_%",
    "Carbon_yield_or_rate",
    "Stability_h",
    "Characterization_methods",
    "Mechanism_keyword",
]
DOC_FIELDS_MECH = ["Active_species", "Proposed_mechanism", "Evidence_methods", "Limitation_keyword"]


def _fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    return None if s in {"", "Not reported", "nan"} else s


def build_corpus(core, mech):
    core = core.dropna(subset=["Record_ID"]).copy()
    mech = mech.dropna(subset=["Record_ID"]).copy()
    merged = core.merge(mech[["Record_ID"] + [c for c in DOC_FIELDS_MECH if c in mech.columns]], on="Record_ID", how="left")
    docs = []
    for _, r in merged.iterrows():
        lines = []
        for f in DOC_FIELDS_CORE + DOC_FIELDS_MECH:
            if f in r.index:
                v = _fmt(r[f])
                if v:
                    lines.append(f"{f}: {v}")
        link = _fmt(r.get("DOI_or_URL"))
        docs.append(
            {
                "Record_ID": int(r["Record_ID"]) if str(r["Record_ID"]).isdigit() else r["Record_ID"],
                "Title": _fmt(r.get("Paper_title")) or "Untitled",
                "Year": _fmt(r.get("Year")),
                "Link": link if link and link.startswith("http") else None,
                "Text": "\n".join(lines),
            }
        )
    return pd.DataFrame(docs)


class Retriever:
    def __init__(self, corpus):
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.corpus = corpus.reset_index(drop=True)
        self._hstack = hstack
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        texts = self.corpus["Text"].fillna("")
        self.matrix = hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def search(self, query, k=4):
        q = self._hstack([self.word.transform([query]), self.char.transform([query])]).tocsr()
        # rows are L2-normalised per block, so the dot product is a (2x-scaled) cosine similarity
        scores = (self.matrix @ q.T).toarray().ravel() / 2.0
        order = np.argsort(-scores)[:k]
        hits = self.corpus.iloc[order].copy()
        hits["Score"] = scores[order]
        return hits[hits["Score"] > 0]


def build_prompt(question, hits, history=None):
    evidence = "\n\n".join(f"[R{h.Record_ID}] {h.Title} ({h.Year or 'n.d.'})\n{h.Text}" for h in hits.itertuples())
    past = ""
    if history:
        turns = [f"{m['role']}: {m['content']}" for m in history[-4:]]
        past = "Recent conversation:\n" + "\n".join(turns) + "\n\n"
    return f"""You are a research assistant for liquid-metal-mediated CO2 conversion.
Answer in the same language as the question (Korean or English).

Rules:
- Use ONLY the evidence records below. If they do not answer the question, say so plainly.
- Cite every claim with its record tag, e.g. [R7]. Do not invent numbers, conditions or references.
- Separate what the papers report from your own interpretation, and label the interpretation.
- Keep it concise: a short answer, then bullet points with citations.

{past}Question: {question}

Evidence records:
{evidence}
""".strip()
