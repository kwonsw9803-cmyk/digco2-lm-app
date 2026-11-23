# DigCO2-LM

An AI-enabled digital catalysis platform for liquid-metal-mediated CO₂ conversion, built with Streamlit.
The interface and its evidence-level structure (reported, computed, modelled, AI-written, tested) are our own.
Three methods were informed by DigMethpy (Cheng et al., *AI Agent* 2026, 2, 8; DOI 10.20517/aiagent.2026.11): the
element map (after Fig. 4B), the regression of computed properties (after Fig. 4A), and recording agreement between a
submitted experiment and the model (after Step 3). Version 0.4 and earlier also followed its platform design.

## Pages

| Evidence level | Page | Path | What it does |
|---|---|---|---|
| | Overview | `/` | Reaction atlas: every record by reaction mode and product class |
| | Evidence levels | `/evidence` | What each level can and cannot support |
| Reported | Records | `/records` | 24 curated literature records with source links, conditions and products per record, plus curated experimental feedback |
| Reported | Ask the records | `/ask` | TF-IDF retrieval over the records + optional Claude answer citing records as [R#] |
| Reported | Element properties | `/elements` | Tabulated element descriptors |
| Computed | Computed properties | `/computed` | DFT/AIMD records (`data/computational_data.csv`), session-only CSV upload, LOOCV regression |
| Modelled | Element map | `/element-map` | Descriptor map and similarity ranking of untested additives |
| Modelled | Product class | `/product-class` | Preliminary classifier, shown against a majority-class baseline |
| Modelled | Candidates | `/candidates` | Pre-computed candidate ranking (Step 6) |
| AI-written | Stored reports | `/stored-reports` | Reports written by Gemini |
| AI-written | New report | `/new-report` | Live Claude report per candidate (off until connected) |
| Tested | Submit a result | `/submit` | Submit a result, compare with the model, save as pending for review |
| | About and references | `/about` | Method credits, data sources, state of each part |

The menu and theme are in `digco2/ui.py` and `.streamlit/config.toml`.

## Secrets (Streamlit Cloud → App settings → Secrets)

```toml
CLAUDE_ENABLED = "0"             # "1" connects Claude; with "0" (default) no call is made even if a key is present
ANTHROPIC_API_KEY = "..."        # needed only when CLAUDE_ENABLED = "1"; never commit it
CLAUDE_MODEL = "claude-opus-5-5" # optional; also claude-sonnet-5-5 or claude-haiku-5-5
CLAUDE_DAILY_LIMIT = "30"        # optional; calls per UTC day for the whole app
DIGCO2_ADMIN = "0"               # "1" shows File Status and pending feedback (do not enable on the public app)
```

Claude is **off by default**. When connected, live calls are limited to 5 per browser session and
`CLAUDE_DAILY_LIMIT` per day for the whole app. The daily counter lives in memory and restarts with the app, so
set a spend limit in the Anthropic Console as well: that is the only hard cap. API usage is billed separately from
Claude Pro/Max subscriptions.

## Data files

- Literature records live in `Final_LM_CO2_platform_database_with_descriptors.xlsx` (Core_database,
  Normalized_database, Mechanism_notes) and `Step3_LM_CO2_database_analysis.xlsx` (Analyzed_database and its count
  sheets). Add records with `python scripts/add_literature_records.py <additions.json>`: it appends to every sheet,
  refreshes the derived sheets, and skips DOIs already present. Records 19–24 were added on 2026-11-20 from
  abstracts (`data/review/literature_additions_2026-11-20.json`; screening of all candidates in
  `data/review/literature_candidates_2026-11-20.csv`). `Step4_ML_ready_dataset.xlsx` and the model still use the
  original 18 records: retraining on 24 was evaluated on 2026-11-23 and 2026-11-24 and not adopted (see Model note).
- The same workbook holds `Conditions` (58 rows, one per record and condition) and `Products` (66 rows, one per condition
  and product), read from the main text of the papers with page numbers. Rules: a number is entered only if the authors
  state it; `Evidence` says whether it was measured, obtained by difference, estimated, shown only in a figure or merely
  mentioned; a product that was not measured is not zero; sums are not forced to 100. `Main_CO2_product` is named only
  when no unmeasured part of the charge could be larger (`digco2/conditions.py`). Records 17 and 18 (thesis chapters
  without a PDF) carry database values only. `python scripts/apply_conditions.py 2026-11-24` rebuilds both sheets from
  the reviewed tables in `data/review` and applies `record_corrections_2026-11-24.csv` (values of records 1, 7, 11, 12,
  14, 16 and 24, and a note on records 19-24) and `record_entries_2026-11-24.csv`.
- `Product_class` is the main product. Where it depends on the condition a record has several entries (column `Entry`) in
  Normalized_database, Electrochemical_subset and Analyzed_database: record 14 is 14-1 (no water, CO) and 14-2 (2 M water,
  mixed C1 products). Core_database and Mechanism_notes keep one row per record. Record 12 stays solid carbon although only
  5.6% of the charge is accounted for: it is the only product measured. `Step4_ML_ready_dataset.xlsx` and the model were not
  changed: the model was trained on the onset potentials of records 11 and 12 and on record 14 as solid carbon.
- `data/computational_data.csv`: one computed property per row (`Solute`, `Property`, `Value`, `Unit`, ...).
  Download the template from the Computed properties page. Only published or public data belongs here.
  The current rows are 0 K bulk formation energies from the Open Quantum Materials Database (CC BY 4.0),
  one convex-hull phase per element and property, each row linked to its OQMD entry:
  - `Oxide_formation_energy_per_O` (eV/O atom): oxygen-scavenging strength
  - `Carbide_formation_energy` (eV/atom; 0 = no stable carbide): carbide versus free-carbon tendency
  - `Ga_X_compound_formation_energy` (eV/atom; 0 = no stable compound): solute–Ga interaction proxy

  These describe bulk solids, not the molten alloy. Cite OQMD as Saal et al., *JOM* 2013, 65, 1501
  (DOI 10.1007/s11837-013-0755-4) and Kirklin et al., *npj Comput. Mater.* 2015, 1, 15010
  (DOI 10.1038/npjcompumats.2015.10).
- `data/feedback/pending_feedback.csv`: written by the feedback form (git-ignored; temporary on Streamlit Cloud).
- `data/feedback/curated_feedback.csv`: reviewed feedback; these rows appear in the Database page.

## Charts

Bar charts use `st.bar_chart`; the screening map and the parity plot use Altair (`digco2/plots.py`),
which ships with Streamlit, so no extra dependency is needed.

## Model note

`product_classifier_random_forest.pkl` is saved with scikit-learn 1.9.1 (pinned in `requirements.txt`).
The original pickle was made with scikit-learn 1.3.2 and stopped working on the Python 3.14 /
scikit-learn 1.9 runtime of Streamlit Cloud. `scripts/migrate_model.py` carried the fitted forest over
unchanged: predicted probabilities are identical to the original model on the 18 training rows and on
5,202 reaction × metal × additive × temperature combinations, so the Step 6 / Step 7 results still match.

When upgrading scikit-learn, run `python scripts/migrate_model.py` with the new version, check the
predictions, then update the pin. `scripts/train_classifier.py` retrains from scratch with the same settings,
but random-forest sampling differs between versions, so its predictions differ from the published model.

### Validation

`scripts/evaluate_classifier.py data/model_evaluation.json` (about half an hour) compares the classifier with the
majority-class baseline; the app shows the saved report. Result on 2026-11-24 with the current classes,
leave-one-paper-out, median of 10 seeds, in entries (record 14 has two):

| Data | Baseline | Original recipe |
|---|---|---|
| 18 records (19 entries) | 12/19 | 7/19 |
| 24 records (25 entries; retrained, not deployed) | 14/25 | 10/25 |
| 6 newer records, deployed model | 2/6 | 2/6 |

With the classes of 2026-11-23 (record 14 as solid carbon) it was 13/18 against 8/18 and 15/24 against 12/24.
No recipe beats the baseline, so the deployed model was kept. The LOOCV accuracy of 0.667 stored in Step4
`Model_summary` is reproduced only without class weights; the deployed settings give 0.50 on the same 18 rows.
