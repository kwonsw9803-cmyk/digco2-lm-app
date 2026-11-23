# DigCO2-LM

An AI-enabled digital catalysis platform for liquid-metal-mediated CO₂ conversion, built with Streamlit.
The platform design follows DigMethpy (Cheng et al., *AI Agent* 2026, 2, 8; DOI 10.20517/aiagent.2026.11):
data → model → prediction → validation → feedback.

## Modules

| Page | What it does |
|---|---|
| Database | 24 curated literature records with source links, plus curated experimental feedback |
| Descriptor Explorer / Screening | Element descriptors; descriptor map and similarity ranking of untested additives |
| Computational Data | DFT/AIMD records (`data/computational_data.csv`), session-only CSV upload, LOOCV descriptor regression |
| ML Prediction | Preliminary product-class classifier, shown against a majority-class baseline |
| Recommendation Agent | Pre-computed candidate ranking (Step 6) |
| Literature Agent | TF-IDF retrieval over the records + optional Claude answer citing records as [R#] |
| Gemini Agent Report / Live Claude Generator | Stored reports (written by Gemini) / live Claude reports per candidate |
| Experimental Feedback | Submit a result, compare with the model, save as pending for review |

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
  original 18 records: retraining on 24 was evaluated on 2026-11-23 and not adopted (see Model note).
- `data/computational_data.csv`: one computed property per row (`Solute`, `Property`, `Value`, `Unit`, ...).
  Download the template from the Computational Data page. Only published or public data belongs here.
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
majority-class baseline; the app shows the saved report. Result on 2026-11-23, leave-one-paper-out, median of 10 seeds:

| Data | Baseline | Original recipe |
|---|---|---|
| 18 records | 13/18 | 8/18 |
| 24 records (retrained, not deployed) | 15/24 | 12/24 |
| 6 newer records, deployed model | 2/6 | 2/6 |

No recipe beats the baseline, so the deployed model was kept. The LOOCV accuracy of 0.667 stored in Step4
`Model_summary` is reproduced only without class weights; the deployed settings give 0.50 on the same 18 rows.
