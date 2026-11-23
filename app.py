import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import digco2

# A push to Streamlit Cloud replaces files under a running process, which keeps the modules it has already imported.
# Drop ours when one of their files is newer than the import, so that app.py never runs against old modules.
if max(f.stat().st_mtime for f in Path(__file__).parent.glob("digco2/*.py")) > getattr(digco2, "LOADED_AT", 0):
    for name in [m for m in sys.modules if m == "digco2" or m.startswith("digco2.")]:
        sys.modules.pop(name, None)

from digco2 import claude_client
from digco2 import computational as comp_mod
from digco2 import conditions as cond_mod
from digco2 import feedback as fb_mod
from digco2 import literature_rag as rag_mod
from digco2 import screening as scr_mod
from digco2 import plots as plot_mod
from digco2 import ui


# ============================================================
# DigCO2-LM Streamlit App
# Interface: digco2/ui.py (pages grouped by evidence level)
# ============================================================

st.set_page_config(
    page_title="DigCO2-LM",
    page_icon=str(Path(__file__).parent / "assets" / "mark.svg"),
    layout="wide",
    initial_sidebar_state="auto",
)

base_dir = Path(__file__).parent

APP_VERSION = "0.6.0"

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

step3_file = base_dir / "Step3_LM_CO2_database_analysis.xlsx"
step4_file = base_dir / "Step4_ML_ready_dataset.xlsx"
step6_file = base_dir / "Step6_agent_recommendation_results.xlsx"
step7b_file = base_dir / "Step7b_Gemini_LLM_agent_report.xlsx"
descriptor_file = base_dir / "Final_LM_CO2_platform_database_with_descriptors.xlsx"
model_file = base_dir / "product_classifier_random_forest.pkl"

data_dir = base_dir / "data"
evaluation_file = data_dir / "model_evaluation.json"
computational_file = data_dir / "computational_data.csv"
pending_feedback_file = data_dir / "feedback" / "pending_feedback.csv"
curated_feedback_file = data_dir / "feedback" / "curated_feedback.csv"


# ============================================================
# Styling
# ============================================================

st.markdown(ui.CSS, unsafe_allow_html=True)
st.logo(str(base_dir / "assets" / "logo.svg"), icon_image=str(base_dir / "assets" / "mark.svg"), size="large")


# ============================================================
# Helper functions
# ============================================================

def file_stamp(path):
    """(mtime, size) of a file. Part of every cache key below, so a redeploy that changes a data file
    reloads it instead of serving the copy cached by the still-running app."""
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data
def _read_excel_sheet(file_path, sheet_name, stamp):
    return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")


def load_excel_sheet(file_path, sheet_name):
    return _read_excel_sheet(str(file_path), sheet_name, file_stamp(file_path))


@st.cache_resource
def _load_model(model_path, stamp):
    return joblib.load(model_path)


def load_model(model_path):
    return _load_model(str(model_path), file_stamp(model_path))


@st.cache_data
def _read_json(path, stamp):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_evaluation():
    """Independent evaluation written by scripts/evaluate_classifier.py, or None."""
    if not evaluation_file.exists():
        return None
    return _read_json(str(evaluation_file), file_stamp(evaluation_file))


def render_unvalidated_warning(extra=""):
    """One-paragraph warning for every place that shows a classifier output."""
    ev = load_model_evaluation()
    if not ev:
        return
    cv = ev["18 original records | leave-one-paper-out | A"]
    base = ev["18 original records | leave-one-paper-out | baseline"]
    new = ev["deployed_on_new_records"]
    hits = sum(r["actual"] == r["predicted"] for r in new)
    k, b = cv["correct_median"], base["correct_median"]
    versus = "fewer than" if k < b else "as many as" if k == b else "more than"
    st.warning(
        f"**The classifier is not validated.** In leave-one-paper-out cross-validation it gets "
        f"{k:g} of {cv['n']} entries right, {versus} always answering 'solid carbon' "
        f"({b:g} of {base['n']}), and {hits} of {len(new)} newer records it had never seen. "
        "Probabilities are model scores, not chances of success." + (" " + extra if extra else "")
    )


def file_status(path):
    return "✅ Found" if path.exists() else "❌ Missing"


def arrow_safe(df):
    """Cast mixed-type text columns (e.g. numbers plus 'Not applicable') to str so Arrow can serialize them."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df


def safe_display(df, height=500, column_config=None):
    st.dataframe(arrow_safe(df), width="stretch", height=height, column_config=column_config)


def get_setting(name, default=None):
    """Read a value from Streamlit secrets first, then from environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


ADMIN_MODE = get_setting("DIGCO2_ADMIN", "0") == "1"
CLAUDE = claude_client.Settings.load(get_setting)
LIVE_CLAUDE_LIMIT_PER_SESSION = 5
CLAUDE_NOT_CONNECTED = (
    "Claude is not connected on this app yet, so nothing is sent to the Claude API. "
)
OTHER_LM_LABEL = "Other (not in training data)"


def training_categories(column):
    """Category values the classifier actually saw during training (Step4 ML_ready_dataset)."""
    if not step4_file.exists():
        return []
    df = load_excel_sheet(step4_file, "ML_ready_dataset")
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).str.strip().unique())


def render_independent_checks(ev):
    """Cross-validation and unseen-record checks, including the 24-record retraining that was not deployed."""

    def row(test, key, rec):
        s, b = ev[f"{key} | {rec}"], ev[f"{key} | baseline"]
        lo, hi = s["wilson95_of_median"]
        return {
            "Test": test,
            "Classifier": f"{s['correct_median']:g} of {s['n']}",
            "Always 'solid carbon'": f"{b['correct_median']:g} of {b['n']}",
            "95% interval of classifier accuracy": f"{lo:.0%} to {hi:.0%}",
        }

    new = ev["deployed_on_new_records"]
    rows = [
        row("Deployed recipe, 18 original records, leave-one-paper-out", "18 original records | leave-one-paper-out", "A"),
        {
            "Test": f"Deployed model on {len(new)} newer records it never saw",
            "Classifier": f"{sum(r['actual'] == r['predicted'] for r in new)} of {len(new)}",
            "Always 'solid carbon'": f"{sum(r['actual'] == 'solid carbon' for r in new)} of {len(new)}",
            "95% interval of classifier accuracy": "too few records",
        },
        row("Retrained on 24 records (not deployed), leave-one-paper-out", "24 records | leave-one-paper-out", "A"),
    ]
    st.markdown("**Independent checks** (median of 10 random seeds)")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    i, s6 = ev["inputs"], ev["step6_if_retrained"]
    binary = ev["binary | leave-one-paper-out | A"]
    n = i["records"]
    changed = ev.get("reclassified_after_training", [])
    reclassified = "".join(
        f"- Record {r['Record_ID']} was '{r['trained_as']}' when the model was trained. The records now list it as "
        + " and ".join(f"'{c}' for {e}" for e, c in r["now"].items())
        + ". The checks above use the new classes and count each entry once; the deployed model was not retrained.\n"
        for r in changed
    )
    onset = ev.get("corrected_after_training", {}).get("Voltage_V", [])
    corrected = (
        f"For records {' and '.join(str(r) for r in onset)} the model was trained on the onset potential; the records now list "
        "the applied potential. " if onset else ""
    )
    st.markdown("**Known limits**")
    st.markdown(
        f"- Retraining on {n} records did not beat the baseline either, so the deployed model still uses the original 18.\n"
        f"- For 'solid carbon as the class or not', the retrained recipe gets {binary['correct_median']:g} of {binary['n']}, but the "
        f"single rule 'electrochemical = no, anything else = yes' gets {ev['binary_single_rule_correct']} of {binary['n']} (rule read "
        "off the same data). Nothing here shows that the model has learned additive chemistry.\n"
        f"{reclassified}"
        f"- {i['distinct_liquid_metal_names']} of {n} liquid-metal names are unique, so the name carries almost no information.\n"
        f"- Voltage applies only to the {i['electrochemical_records']} electrochemical records. {i['reported_voltage']} of them have a "
        f"single number in the model inputs, against different reference electrodes. {corrected}Current density "
        f"is available there for {i['reported_current_density']} of {i['electrochemical_records']} (with mixed signs) and a single "
        f"temperature for {i['reported_temperature']} of {n}. Missing values are filled with the median, which also gives a "
        "voltage to records that have none, so these inputs barely inform the prediction.\n"
        "- Classes with a single record cannot be predicted when that record is left out.\n"
        f"- Step 6 ranking is sensitive to the training data: with the {n}-record model only {s6['top20_overlap']} of the top 20 "
        f"rows would stay (additives {', '.join(s6['top20_additives'][0])} would become {', '.join(s6['top20_additives'][1])}; "
        "many rows are tied)."
    )


def render_model_performance():
    """Compare the classifier with a majority-class baseline, using Step4 outputs."""
    if not step4_file.exists():
        return
    xls = pd.ExcelFile(step4_file, engine="openpyxl")

    with st.expander("Model performance vs. majority-class baseline", expanded=True):
        ev = load_model_evaluation()
        data = load_excel_sheet(step4_file, "ML_ready_dataset")
        if "Product_class" not in data.columns or data.empty:
            st.info("Product_class column not found in the ML-ready dataset.")
            return

        counts = data["Product_class"].fillna("Not reported").value_counts()
        majority_class = counts.index[0]
        baseline_acc = counts.iloc[0] / counts.sum()

        rows = []
        if "Model_summary" in xls.sheet_names:
            summary = load_excel_sheet(step4_file, "Model_summary")
            for _, r in summary.iterrows():
                note = r.get("Comment", "")
                if ev and str(r["Model"]) == "RandomForestClassifier_LOOCV":
                    s = ev["loocv_18_published_settings"]
                    if abs(s["without_class_weight"] - float(r["Accuracy_or_CV_score"])) < 0.005:
                        note = (
                            f"Reproduced only without class weights; the deployed model uses balanced class "
                            f"weights and scores {s['deployed_settings_balanced']:.3f}"
                        )
                rows.append({"Model": r["Model"], "Accuracy": r["Accuracy_or_CV_score"], "Note": note})
        rows.append(
            {
                "Model": f"Majority-class baseline (always '{majority_class}')",
                "Accuracy": baseline_acc,
                "Note": f"{counts.iloc[0]} of {counts.sum()} records are '{majority_class}'",
            }
        )
        perf = pd.DataFrame(rows)

        best = perf.loc[~perf["Model"].str.startswith("Majority"), "Accuracy"].max()
        if pd.notna(best) and best <= baseline_acc + 1e-9:
            st.warning(
                f"No trained model beats the majority-class baseline ({baseline_acc:.3f}). "
                f"With {counts.sum()} records the classifier mostly predicts '{majority_class}', "
                "so treat its output as a placeholder until more data are added."
            )

        st.markdown("**Accuracy**")
        st.dataframe(
            perf,
            width="stretch",
            hide_index=True,
            column_config={"Accuracy": st.column_config.NumberColumn(format="%.3f")},
        )
        if ev:
            render_independent_checks(ev)

        st.markdown("**Class distribution (training data)**")
        dist = counts.rename_axis("Product_class").reset_index(name="Records")
        st.dataframe(dist, width="stretch", hide_index=True)

        if "Prediction_results" in xls.sheet_names:
            pr = load_excel_sheet(step4_file, "Prediction_results")
            rf = pr[pr["Model"].astype(str).str.contains("RandomForest", case=False)]
            if not rf.empty:
                st.markdown(f"**Confusion matrix — RandomForest hold-out set ({len(rf)} records)**")
                cm = pd.crosstab(
                    rf["Actual"].rename("Actual"),
                    rf["Predicted"].rename("Predicted"),
                )
                st.dataframe(cm, width="stretch")


def render_footer():
    st.divider()
    st.caption(
        f"DigCO2-LM {APP_VERSION}. For research and education. Modelled and AI-written content is not validated "
        "by experiment. Sources and method credits are on the About and references page."
    )


@st.cache_resource
def claude_daily_limit():
    """Shared by every visitor of this app process (a page refresh does not reset it)."""
    try:
        limit = int(get_setting("CLAUDE_DAILY_LIMIT", "30"))
    except ValueError:
        limit = 30
    return claude_client.DailyLimit(max(limit, 0))


def claude_block_reason():
    """Why a live Claude call is not possible right now, or None."""
    problem = CLAUDE.problem()
    if problem:
        # Configuration details are for the maintainer; visitors only see that Claude is off.
        return problem if ADMIN_MODE else claude_client.OFF
    if st.session_state.get("live_claude_calls", 0) >= LIVE_CLAUDE_LIMIT_PER_SESSION:
        return "The live Claude limit for this session has been reached."
    if claude_daily_limit().remaining() <= 0:
        return "The app's daily Claude limit has been reached. Please come back tomorrow."
    return None


def call_claude(prompt):
    """One live call, counted against the session and the app-wide daily limit."""
    reason = claude_block_reason()
    if reason:
        return claude_client.Reply(False, reason)
    if not claude_daily_limit().take():
        return claude_client.Reply(False, "The app's daily Claude limit has been reached. Please come back tomorrow.")
    st.session_state["live_claude_calls"] = st.session_state.get("live_claude_calls", 0) + 1
    return claude_client.ask(prompt, CLAUDE)


def normalize_descriptor_row(desc_df, additive):
    row = desc_df[desc_df["Element"].astype(str).str.strip() == additive]
    if row.empty:
        return {}

    r = row.iloc[0]

    def g(col, default="Not reported"):
        if col in r.index and pd.notna(r[col]):
            v = r[col]
            return v.item() if hasattr(v, "item") else v  # numpy scalar -> Python (st.json shows np.int64 as text)
        return default

    return {
        "Additive_Element_group": g("Element_group"),
        "Additive_Common_oxidation_states": g("Common_oxidation_states"),
        "Additive_Oxide_formula": g("Oxide_formula"),
        "Additive_Oxide_formation_tendency": g("Oxide_formation_tendency"),
        "Additive_Oxygen_affinity_class": g("Oxygen_affinity_class"),
        "Additive_Reductant_strength_class": g("Reductant_strength_class"),
        "Additive_CO2_to_solid_carbon_relevance": g("CO2_to_solid_carbon_relevance"),
        "Additive_Expected_role_in_LM_CO2": g("Expected_role_in_LM_CO2"),
        "Additive_Period": g("Period", 0),
        "Additive_Group_number": g("Group_number", 0),
        "Additive_Atomic_number": g("Atomic_number", 0),
        "Additive_Atomic_weight": g("Atomic_weight", 0),
        "Additive_Melting_point_C": g("Melting_point_C", 0),
        "Additive_Boiling_point_C": g("Boiling_point_C", 0),
        "Additive_Electronegativity_Pauling": g("Electronegativity_Pauling", 0),
        "Additive_Atomic_radius_pm": g("Atomic_radius_pm", 0),
        "Additive_Standard_reduction_potential_V_vs_SHE": g("Standard_reduction_potential_V_vs_SHE", 0),
    }


def make_condition_for_model(
    reaction_type,
    liquid_metal,
    additive,
    temp_c,
    voltage_v,
    current_density,
    desc_df,
):
    d = normalize_descriptor_row(desc_df, additive)

    condition = {
        "Reaction_type_for_ML": reaction_type,
        "Liquid_metal_clean_for_ML": liquid_metal,
        "Additive_element_for_descriptor": additive,
        "Temp_representative_C_ML": temp_c,
        "Voltage_V_ML": voltage_v,
        "Current_density_mA_cm2_ML": current_density,
        "Is_electrochemical": int("electrochemical" in reaction_type.lower()),
        "Is_thermal": int(("thermal" in reaction_type.lower()) or ("chemical" in reaction_type.lower())),
        "Is_mechanical": int(("mechanical" in reaction_type.lower()) or ("triboelectrochemical" in reaction_type.lower())),
        "Is_plasma": int("plasma" in reaction_type.lower()),
        "Is_sonochemical": int("sono" in reaction_type.lower()),
    }
    condition.update(d)
    return condition


@st.cache_resource
def _build_retriever(stamp):
    core = load_excel_sheet(descriptor_file, "Core_database")
    mech = load_excel_sheet(descriptor_file, "Mechanism_notes")
    return rag_mod.Retriever(rag_mod.build_corpus(core, mech))


def get_retriever():
    return _build_retriever(file_stamp(descriptor_file))


def predict_condition(condition):
    """Run the product classifier on one condition dict. Returns (class, probability table or None)."""
    bundle = load_model(model_file)
    model = bundle["model"]
    categorical_features = bundle["categorical_features"]
    numeric_features = bundle["numeric_features"]
    training_columns = bundle["training_columns"]

    X = pd.DataFrame([condition])
    for col in training_columns:
        if col not in X.columns:
            X[col] = "Not reported" if col in categorical_features else 0
    X = X[training_columns].copy()
    for col in categorical_features:
        X[col] = X[col].fillna("Not reported").astype(str)
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    pred = model.predict(X)[0]
    prob_df = None
    clf = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model
    if clf is not None and hasattr(clf, "predict_proba"):
        prob_df = (
            pd.DataFrame({"Product_class": clf.classes_, "Probability": model.predict_proba(X)[0]})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
    return pred, prob_df


def build_candidate_prompt(row):
    lm = row.get("Liquid_metal_clean_for_ML", "Not reported")
    additive = row.get("Additive_element_for_descriptor", "Not reported")
    reaction = row.get("Reaction_type_for_ML", "Not reported")
    temp = row.get("Temp_representative_C_ML", "Not reported")
    product = row.get("Predicted_Product_class", "Not reported")
    prob = row.get("ML_target_probability", "Not reported")
    score = row.get("Final_recommendation_score", "Not reported")
    reason = row.get("Recommendation_reason", "Not reported")
    support = row.get("Literature_support", "Not reported")

    return f"""
You are a catalysis research agent specialized in liquid-metal-mediated CO2 conversion.

Use Korean language.

Write a concise recommendation report for the following candidate.
Do not invent numerical values.
Clearly distinguish literature/database evidence from model-based prediction.
Mention that this model is preliminary because the dataset is small.
Use a structured format suitable for a research meeting.

Candidate:
- Liquid metal system: {lm}-{additive}
- Reaction type: {reaction}
- Temperature: {temp} °C
- Predicted product class: {product}
- ML target probability: {prob}
- Final recommendation score: {score}

Platform recommendation reason:
{reason}

Literature/database support:
{support}

Please include:
1. Recommendation summary
2. Why this candidate is promising
3. Suggested initial experiment
4. Recommended characterization
5. Key risks and limitations
""".strip()


# ============================================================
# Navigation
# ============================================================

# Server paths and API-key status are only shown when DIGCO2_ADMIN = "1" is set in secrets.
page, PAGES = ui.build_navigation(admin=ADMIN_MODE)


# ============================================================
# Overview
# ============================================================

if page == "Overview":
    ui.page_header("What has liquid metal been shown to make from CO₂?")

    if not step3_file.exists():
        st.warning("Database file is missing: Step3_LM_CO2_database_analysis.xlsx.")
    else:
        df = load_excel_sheet(step3_file, "Analyzed_database")
        links = {}
        if descriptor_file.exists():
            core = load_excel_sheet(descriptor_file, "Core_database")
            links = dict(zip(core["Record_ID"].astype(str), core["DOI_or_URL"].astype(str)))
        comp_df, _ = comp_mod.validate_computational(comp_mod.load_computational(computational_file))
        n_curated = len(fb_mod.load_feedback(curated_feedback_file))
        years = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int)
        span = f", from {years.min()} to {years.max()}" if not years.empty else ""
        split = ui.split_records(df)
        st.markdown(
            f"**{df['Record_ID'].nunique()} published records**{span}. Each number below is one record: select it to open the source. "
            "Rows show how the reaction is driven, columns show the main product class."
            + "".join(
                f" Record {r} appears as {' and '.join(labels)}, because its main product depends on the condition."
                for r, labels in split.items()
            )
        )
        st.markdown(ui.reaction_atlas(df, links), unsafe_allow_html=True)
        single = ui.single_product_modes(df)
        if single:
            st.markdown(
                "In these records, "
                + " and ".join(
                    f"{'both' if n == 2 else f'all {n}'} {mode} records report {product}" for mode, n, product in single
                )
                + "."
            )
        st.caption(
            "An empty cell means no record in this database, not that the product is impossible. "
            "Dashed numbers have no source link. The same records are listed as a table on the Records page."
        )

        st.subheader("Continue from here", anchor=False)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.page_link(PAGES["Literature Agent"], label="**Ask the records**")
            st.markdown("Find which papers speak to a question about mechanism, conditions or product.")
        with c2:
            st.page_link(PAGES["Descriptor Screening"], label="**Look for an untested element**")
            st.markdown("Compare elements nobody has reported with those that already gave a product.")
        with c3:
            st.page_link(PAGES["Experimental Feedback"], label="**Submit a result**")
            st.markdown("Add a lab result. A maintainer reviews it before it joins the records.")
        st.caption(
            f"Besides these records: {len(comp_df)} computed values for {comp_df['Solute'].nunique()} elements, "
            f"and {n_curated} reviewed lab results so far."
        )


# ============================================================
# Evidence levels
# ============================================================

elif page == "Evidence levels":
    ui.page_header(
        "Evidence levels",
        lead="Each content page carries one of five labels. They say where its content comes from and what you may conclude from it.",
    )
    n_lit = load_excel_sheet(step3_file, "Analyzed_database")["Record_ID"].nunique() if step3_file.exists() else 0
    comp_df, _ = comp_mod.validate_computational(comp_mod.load_computational(computational_file))
    n_cand = len(load_excel_sheet(step6_file, "All_candidate_predictions")) if step6_file.exists() else 0
    n_rep = len(load_excel_sheet(step7b_file, "Gemini_candidate_reports")) if step7b_file.exists() else 0
    n_curated = len(fb_mod.load_feedback(curated_feedback_file))
    rows = [
        ("Reported", f"{n_lit} published records, plus tabulated element properties",
         "What a study observed, under its own conditions.",
         "A rate or yield for other conditions. The main rows of records 19 to 24 come from abstracts; their conditions and "
         "products come from the papers.",
         ["Database", "Literature Agent", "Descriptor Explorer"]),
        ("Computed", f"{len(comp_df)} values from the Open Quantum Materials Database",
         "Relative bulk stability at 0 K, for example how strongly an element holds oxygen.",
         "Behaviour in the molten alloy.",
         ["Computational Data"]),
        ("Modelled", f"1 classifier and {n_cand} scored candidates",
         "Which experiments could be worth a look.",
         "Which experiment will work. The classifier does not beat always answering 'solid carbon'.",
         ["Descriptor Screening", "ML Prediction", "Recommendation Agent"]),
        ("AI-written", f"{n_rep} stored reports",
         "A readable draft to check against the papers.",
         "A source. Do not cite it; cite the papers.",
         ["Gemini Agent Report", "Live Claude Generator"]),
        ("Tested", f"{n_curated} reviewed lab results",
         "What a user measured, once reviewed.",
         "Anything while it is still pending review.",
         ["Experimental Feedback"]),
    ]
    for level, amount, yes, no, keys in rows:
        st.divider()
        c1, c2, c3 = st.columns([1.1, 2.4, 1.1])
        with c1:
            st.subheader(level, anchor=False)
            st.caption(amount)
        with c2:
            st.markdown(f"**You can read from it:** {yes}")
            st.markdown(f"**You cannot read from it:** {no}")
        with c3:
            for k in keys:
                st.page_link(PAGES[k])


# ============================================================
# About and references
# ============================================================

elif page == "About":
    ui.page_header("About and references")
    st.markdown(
        "DigCO2-LM collects published results on CO₂ conversion with liquid metals and keeps them apart from "
        "anything computed, modelled or written by AI. It is for research and education."
    )

    st.subheader("Method credits", anchor=False)
    st.markdown(
        "The interface and the evidence-level structure are our own. Three methods were informed by DigMethpy "
        "(Cheng, Z. et al. *AI Agent* 2026, 2, 8, "
        "[DOI 10.20517/aiagent.2026.11](https://doi.org/10.20517/aiagent.2026.11)):\n"
        "- the two-descriptor element map (after its Fig. 4B)\n"
        "- the regression of computed properties on elemental features (after its Fig. 4A)\n"
        "- recording whether a submitted experiment agrees with the model (after its Step 3)\n\n"
        "Version 0.4 and earlier also followed DigMethpy's overall platform design."
    )

    st.subheader("Data sources", anchor=False)
    st.markdown(
        "- Literature records: the papers and the thesis linked on the Records page.\n"
        "- Computed values: Open Quantum Materials Database, CC BY 4.0. Saal et al., *JOM* 2013, 65, 1501, "
        "[DOI 10.1007/s11837-013-0755-4](https://doi.org/10.1007/s11837-013-0755-4); Kirklin et al., "
        "*npj Comput. Mater.* 2015, 1, 15010, "
        "[DOI 10.1038/npjcompumats.2015.10](https://doi.org/10.1038/npjcompumats.2015.10)."
    )

    st.subheader("State of each part", anchor=False)
    st.dataframe(
        pd.DataFrame(
            [
                ("Literature records", "Available", "24 curated records with source links"),
                ("Conditions and products", "Available", "Read from the main text, with pages; supporting information not read"),
                ("Element properties and map", "Available", "Similarity heuristic, not an activity model"),
                ("Computed properties", "Available", "Public bulk data only; session upload available"),
                ("Regression of computed properties", "Available when data exist", "Needs at least 5 solutes per property"),
                ("Product classifier", "Preliminary", "Does not beat the majority-class baseline"),
                ("Ask the records", "Available", "Retrieves records; Claude answers are off until connected"),
                ("Experiment submissions", "Available", "Pending, then maintainer review, then curated"),
                ("Automated literature mining", "Planned", "PDF parsing and extraction with quality checks"),
                ("Machine-learned interatomic potentials", "Planned", "For sampling molten surfaces"),
            ],
            columns=["Part", "State", "Note"],
        ),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# Database
# ============================================================

elif page == "Database":
    ui.page_header("Records", "Reported", "Published liquid-metal CO₂ conversion results, one row per record.")

    if not step3_file.exists():
        st.warning("Step3 database file not found.")
    else:
        df = load_excel_sheet(step3_file, "Analyzed_database").assign(Origin="Literature")
        curated = fb_mod.curated_as_database_rows(fb_mod.load_feedback(curated_feedback_file))
        if not curated.empty:
            df = pd.concat([df, curated.assign(Origin="Experimental feedback (curated)")], ignore_index=True)
            st.caption(f"Includes {len(curated)} curated experimental-feedback record(s).")
        filtered = df.copy()

        c1, c2, c3 = st.columns(3)

        with c1:
            if "Reaction_type_clean" in filtered.columns:
                selected_reactions = st.multiselect(
                    "Reaction type",
                    sorted(filtered["Reaction_type_clean"].dropna().unique()),
                )
                if selected_reactions:
                    filtered = filtered[filtered["Reaction_type_clean"].isin(selected_reactions)]

        with c2:
            if "Product_class" in filtered.columns:
                selected_products = st.multiselect(
                    "Product class",
                    sorted(filtered["Product_class"].dropna().unique()),
                )
                if selected_products:
                    filtered = filtered[filtered["Product_class"].isin(selected_products)]

        with c3:
            if "Additive_metal_clean" in filtered.columns:
                selected_additives = st.multiselect(
                    "Additive metal",
                    sorted(filtered["Additive_metal_clean"].dropna().astype(str).unique()),
                )
                if selected_additives:
                    filtered = filtered[filtered["Additive_metal_clean"].astype(str).isin(selected_additives)]

        df = df.assign(Entry=df.apply(ui.entry_label, axis=1))
        filtered = filtered.assign(Entry=filtered.apply(ui.entry_label, axis=1) if len(filtered) else [])
        st.write(f"Filtered entries: {len(filtered)} / {len(df)}")
        for r, labels in ui.split_records(df).items():
            st.caption(f"Record {r} has {len(labels)} entries ({', '.join(labels)}), because its main product depends on the condition.")

        # Attach source links from the curated Core_database (same Record_ID).
        if descriptor_file.exists() and "Record_ID" in filtered.columns:
            core = load_excel_sheet(descriptor_file, "Core_database")
            if {"Record_ID", "DOI_or_URL"} <= set(core.columns):
                links = core[["Record_ID", "DOI_or_URL"]].dropna(subset=["Record_ID"])
                links = links.assign(
                    Record_ID=links["Record_ID"].astype(str),
                    Source_link=links["DOI_or_URL"].where(
                        links["DOI_or_URL"].astype(str).str.startswith("http"), None
                    ),
                )[["Record_ID", "Source_link"]]
                filtered = filtered.assign(Record_ID=filtered["Record_ID"].astype(str)).merge(
                    links, on="Record_ID", how="left"
                )
                if "Source" in filtered.columns:  # feedback rows carry their own Source text
                    is_url = filtered["Source"].astype(str).str.startswith("http")
                    filtered["Source_link"] = filtered["Source_link"].fillna(filtered["Source"].where(is_url))
                filtered["Source"] = filtered["Source_link"]

        key_cols = [
            "Entry",
            "Origin",
            "Paper_title",
            "Source",
            "Year",
            "LM_composition",
            "Base_LM",
            "Additive_metal_clean",
            "Reaction_type_clean",
            "Temp_representative_C",
            "Main_product",
            "Product_class",
            "CO2_conversion_representative_%",
            "Carbon_rate_umol_h",
            "FE_total_%",
            "Mechanism_keyword",
        ]
        key_cols = [c for c in key_cols if c in filtered.columns]

        safe_display(
            filtered[key_cols],
            height=600,
            column_config={"Source": st.column_config.LinkColumn("Source", display_text="Open")},
        )

        st.download_button(
            "Download filtered database as CSV",
            filtered.to_csv(index=False, encoding="utf-8-sig"),
            file_name="DigCO2_LM_filtered_database.csv",
            mime="text/csv",
        )

        if descriptor_file.exists():
            conds = load_excel_sheet(descriptor_file, "Conditions")
            prods = load_excel_sheet(descriptor_file, "Products")
            st.subheader("Conditions and products of one record", anchor=False)
            st.markdown(
                f"A paper often reports several conditions. The database holds **{len(conds)} conditions** and "
                f"**{len(prods)} product entries**, read from the papers and listed with their page."
            )
            lit_ids = pd.to_numeric(filtered.loc[filtered["Origin"] == "Literature", "Record_ID"], errors="coerce").dropna()
            lit_ids = lit_ids.astype(int).drop_duplicates().tolist()
            lit = df[df["Origin"] == "Literature"]
            lit = lit.set_index(pd.to_numeric(lit["Record_ID"], errors="coerce").astype(int))
            titles = lit["Paper_title"][~lit.index.duplicated()]
            n_cond, n_prod = conds["Record_ID"].value_counts(), prods["Record_ID"].value_counts()
            if not lit_ids:
                st.caption("No literature record matches the filters above.")
            else:
                def record_label(r):
                    c, k = int(n_cond.get(r, 0)), int(n_prod.get(r, 0))
                    return (
                        f"{r}: {titles[r]} ({c} condition{'' if c == 1 else 's'}, "
                        f"{k} product entr{'y' if k == 1 else 'ies'})"
                    )

                rid = st.selectbox("Record", lit_ids, format_func=record_label)
                entries = lit.loc[[rid]]
                if len(entries) == 1:
                    st.markdown(f"Product class in the table above: **{entries['Product_class'].iloc[0]}**")
                else:
                    st.markdown(
                        "Product class in the table above: "
                        + ", ".join(f"**{e['Product_class']}** for {e['Entry']}" for _, e in entries.iterrows())
                    )
                for note in entries["Data_notes"].dropna().astype(str).drop_duplicates():
                    st.caption(note)
                st.markdown("**Conditions**")
                st.dataframe(cond_mod.conditions_view(conds[conds["Record_ID"] == rid]), width="stretch", hide_index=True)
                st.markdown("**Products**")
                if n_prod.get(rid, 0):
                    st.dataframe(cond_mod.products_view(prods[prods["Record_ID"] == rid]), width="stretch", hide_index=True)
                else:
                    st.markdown("No product share was found in the main text of this record.")
            st.caption(
                "An empty cell was not found in the main text; supporting information was not read. Records 17 and 18 have no paper "
                "file, so their rows repeat the main table. A product that was not "
                "measured is not zero, and sums are not forced to 100. A main product is named only when nothing unmeasured "
                "could be larger. These tables are not used by the classifier."
            )
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download all conditions as CSV",
                    conds.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="DigCO2_LM_conditions.csv",
                    mime="text/csv",
                )
            with d2:
                st.download_button(
                    "Download all products as CSV",
                    prods.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="DigCO2_LM_products.csv",
                    mime="text/csv",
                )


# ============================================================
# Descriptor Explorer
# ============================================================

elif page == "Descriptor Explorer":
    ui.page_header("Element properties", "Reported", "Tabulated properties of the elements used as additives or hosts.")

    if not descriptor_file.exists():
        st.warning("Descriptor file not found.")
    else:
        desc = load_excel_sheet(descriptor_file, "Element_descriptors")
        st.write("Descriptor database shape:", desc.shape)

        elements = sorted(desc["Element"].dropna().astype(str).unique())
        selected_elements = st.multiselect(
            "Select elements",
            elements,
            default=elements[:5] if len(elements) >= 5 else elements,
        )

        if selected_elements:
            filtered_desc = desc[desc["Element"].astype(str).isin(selected_elements)]
        else:
            filtered_desc = desc

        safe_display(filtered_desc, height=500)

        st.divider()

        st.subheader("Descriptor distribution")
        c1, c2 = st.columns(2)

        for col, field, title in [
            (c1, "Oxygen_affinity_class", "Oxygen affinity class"),
            (c2, "Reductant_strength_class", "Reductant strength class"),
        ]:
            with col:
                if field in desc.columns:
                    st.markdown(f"**{title}**")
                    st.bar_chart(desc[field].fillna("Not reported").value_counts(), x_label="", y_label="Elements")


# ============================================================
# ML Prediction
# ============================================================

elif page == "ML Prediction":
    ui.page_header("Product class", "Modelled", "A preliminary classifier trained on a small set of published records. Check its performance before using it.")

    if not model_file.exists():
        st.warning("Model file not found. product_classifier_random_forest.pkl is required.")
    elif not descriptor_file.exists():
        st.warning("Descriptor file not found.")
    else:
        bundle = load_model(model_file)
        model = bundle["model"]
        categorical_features = bundle["categorical_features"]
        numeric_features = bundle["numeric_features"]
        training_columns = bundle["training_columns"]

        desc = load_excel_sheet(descriptor_file, "Element_descriptors")

        render_model_performance()

        st.divider()

        # Options come from the categories the classifier was trained on.
        # The encoder uses handle_unknown="ignore", so any other string is silently
        # treated as "no information" and has no effect on the prediction.
        trained_reactions = training_categories("Reaction_type_for_ML") or [
            "thermal", "electrochemical", "mechanical", "membrane-assisted", "plasma", "sonochemical",
        ]
        trained_lms = training_categories("Liquid_metal_clean_for_ML")
        trained_additives = set(training_categories("Additive_element_for_descriptor"))

        c1, c2, c3 = st.columns(3)
        with c1:
            reaction_type = st.selectbox("Reaction type", trained_reactions)
            lm_choice = st.selectbox(
                "Liquid metal (as named in the training data)",
                trained_lms + [OTHER_LM_LABEL],
            )
            if lm_choice == OTHER_LM_LABEL:
                liquid_metal = "Not reported"
                st.caption("An unseen liquid metal carries no information for this model; only the other inputs are used.")
            else:
                liquid_metal = lm_choice

        with c2:
            additive = st.selectbox(
                "Additive metal",
                sorted(desc["Element"].dropna().astype(str).unique()),
                format_func=lambda el: el if el in trained_additives else f"{el} (descriptors only)",
            )
            temp_c = st.number_input("Temperature (°C)", value=40.0, step=10.0)

        with c3:
            sparse = "Reported for few training records (the rest were filled with the median), so it barely informs the prediction."
            voltage_v = st.number_input("Voltage (V)", value=0.0, step=0.1, help=sparse)
            current_density = st.number_input("Current density (mA/cm²)", value=0.0, step=0.1, help=sparse)

        if st.button("Predict product class"):
            condition = make_condition_for_model(
                reaction_type,
                liquid_metal,
                additive,
                temp_c,
                voltage_v,
                current_density,
                desc,
            )

            pred, prob_df = predict_condition(condition)

            st.success(f"Predicted product class: {pred}")
            render_unvalidated_warning()

            if prob_df is not None:
                st.subheader("Prediction probability")
                st.dataframe(prob_df, width="stretch", hide_index=True)
                st.bar_chart(prob_df.set_index("Product_class")["Probability"])

            st.subheader("Descriptor information")
            desc_info = normalize_descriptor_row(desc, additive)
            st.json(desc_info)


# ============================================================
# Recommendation Agent
# ============================================================

elif page == "Recommendation Agent":
    ui.page_header("Candidates", "Modelled", "Combinations scored by the classifier and simple rules.")

    if not step6_file.exists():
        st.warning("Step6_agent_recommendation_results.xlsx not found.")
    else:
        top_df = load_excel_sheet(step6_file, "Top_recommendations")
        all_df = load_excel_sheet(step6_file, "All_candidate_predictions")

        c1, c2, c3 = st.columns(3)
        c1.metric("Top recommendations", len(top_df))
        c2.metric("All candidates", len(all_df))
        if "Predicted_Product_class" in top_df.columns:
            c3.metric("Predicted classes", top_df["Predicted_Product_class"].nunique())

        ev = load_model_evaluation()
        render_unvalidated_warning(
            f"The final score is 10 × ML probability + rule score, and only {ev['step6_if_retrained']['top20_overlap']} of the "
            f"top 20 rows would stay if the model were retrained on {ev['inputs']['records']} records. "
            "Use this list to choose experiments, not as evidence."
            if ev
            else ""
        )

        st.divider()

        show_cols = [
            "Reaction_type_for_ML",
            "Liquid_metal_clean_for_ML",
            "Additive_element_for_descriptor",
            "Temp_representative_C_ML",
            "Predicted_Product_class",
            "ML_target_probability",
            "Rule_based_score",
            "Final_recommendation_score",
            "Recommendation_reason",
            "Literature_support",
        ]
        show_cols = [c for c in show_cols if c in top_df.columns]

        lm_col = "Liquid_metal_clean_for_ML"
        trained_lms = set(training_categories(lm_col))
        if lm_col in top_df.columns and trained_lms:
            unseen = sorted(set(top_df[lm_col].dropna().astype(str)) - trained_lms)
            if unseen:
                st.warning(
                    "These candidates were generated with liquid-metal names the classifier never saw in training "
                    f"({', '.join(unseen)}). The model ignores unseen names, so the liquid-metal column does not "
                    "change the scores below: rows that differ only by liquid metal have identical predictions."
                )

        merge_lm = st.checkbox(
            "Merge candidates that differ only by liquid metal",
            value=True,
            help="Groups rows whose prediction and scores are identical and lists the liquid metals together.",
        )

        view_df = top_df
        if merge_lm and lm_col in top_df.columns:
            group_keys = [
                c
                for c in [
                    "Reaction_type_for_ML",
                    "Additive_element_for_descriptor",
                    "Temp_representative_C_ML",
                    "Predicted_Product_class",
                    "ML_target_probability",
                    "Final_recommendation_score",
                ]
                if c in top_df.columns
            ]
            agg = {c: "first" for c in top_df.columns if c not in group_keys}
            agg[lm_col] = lambda s: ", ".join(dict.fromkeys(s.dropna().astype(str)))
            view_df = (
                top_df.groupby(group_keys, dropna=False, sort=False)
                .agg(agg)
                .reset_index()[top_df.columns]
            )
            st.caption(f"{len(top_df)} rows merged into {len(view_df)} distinct candidates.")

        st.subheader("Top recommendations")
        safe_display(view_df[show_cols], height=500)

        st.download_button(
            "Download recommendation results",
            top_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="DigCO2_LM_top_recommendations.csv",
            mime="text/csv",
        )

        st.divider()

        st.subheader("Candidate detail")
        view_df = view_df.reset_index(drop=True)
        options = [
            f"{i+1}. {row.get('Liquid_metal_clean_for_ML', 'LM')} + {row.get('Additive_element_for_descriptor', 'Additive')} | {row.get('Reaction_type_for_ML', '')}"
            for i, row in view_df.iterrows()
        ]

        if options:
            selected = st.selectbox("Select candidate", options)
            row = view_df.iloc[options.index(selected)]

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Predicted product", row.get("Predicted_Product_class", "N/A"))
                st.metric("ML probability", row.get("ML_target_probability", "N/A"))
                st.metric("Final score", row.get("Final_recommendation_score", "N/A"))

            with c2:
                st.markdown("**Recommendation reason**")
                st.write(row.get("Recommendation_reason", "Not reported"))
                st.markdown("**Literature support**")
                st.write(row.get("Literature_support", "Not reported"))


# ============================================================
# Gemini Agent Report
# ============================================================

elif page == "Gemini Agent Report":
    ui.page_header("Stored reports", "AI-written")

    if not step7b_file.exists():
        st.warning("Step7b_Gemini_LLM_agent_report.xlsx not found.")
    else:
        st.caption("Stored reports, written by Gemini before the app moved to Claude. They are kept as they were generated.")
        render_unvalidated_warning("These stored reports quote those scores.")
        report_df = load_excel_sheet(step7b_file, "Gemini_candidate_reports")
        st.write("LLM reports:", report_df.shape)

        for _, row in report_df.iterrows():
            rank = row.get("Rank", "N/A")
            lm_system = row.get("LM_system", "Unknown")
            report = row.get("Gemini_LLM_report", "No report")

            with st.expander(f"Candidate {rank}: {lm_system}"):
                st.markdown(report)

        xls = pd.ExcelFile(step7b_file, engine="openpyxl")
        if "Overall_synthesis" in xls.sheet_names:
            overall_df = load_excel_sheet(step7b_file, "Overall_synthesis")
            if "Overall_synthesis" in overall_df.columns:
                st.divider()
                st.subheader("Overall synthesis")
                st.markdown(overall_df["Overall_synthesis"].iloc[0])


# ============================================================
# Live Claude Generator
# ============================================================

elif page == "Live Claude Generator":
    ui.page_header("New report", "AI-written", "Draft a report for one candidate with Claude.")

    if CLAUDE.problem():
        st.info(CLAUDE_NOT_CONNECTED + "You can still preview the prompt below.")
    else:
        st.info("Each report calls the Claude API and uses paid API credit.")

    if not step6_file.exists():
        st.warning("Step6 recommendation result file not found.")
    else:
        top_df = load_excel_sheet(step6_file, "Top_recommendations")

        options = [
            f"{i+1}. {row.get('Liquid_metal_clean_for_ML', 'LM')}-{row.get('Additive_element_for_descriptor', 'Additive')} | {row.get('Reaction_type_for_ML', '')}"
            for i, row in top_df.iterrows()
        ]

        if not options:
            st.info("No candidates available.")
        else:
            selected = st.selectbox("Select candidate", options)
            row = top_df.iloc[options.index(selected)]

            render_unvalidated_warning("The prompt below quotes those scores.")
            prompt = build_candidate_prompt(row)
            st.caption(f"Model: {CLAUDE.model}")

            with st.expander("Prompt preview"):
                st.code(prompt)

            blocked = claude_block_reason()
            if st.button("Generate live Claude report", disabled=blocked is not None):
                with st.spinner("Calling the Claude API..."):
                    reply = call_claude(prompt)
                if reply.ok:
                    # Kept for the session so that a later click does not discard a paid report.
                    st.session_state["claude_report"] = (selected, reply)
                else:
                    st.warning(reply.text)

            if not CLAUDE.problem():
                remaining = max(LIVE_CLAUDE_LIMIT_PER_SESSION - st.session_state.get("live_claude_calls", 0), 0)
                st.caption(
                    f"Live Claude calls left in this session: {remaining} / {LIVE_CLAUDE_LIMIT_PER_SESSION} "
                    "(shared with Ask the records)"
                )
                if blocked:
                    st.warning(blocked)
            elif ADMIN_MODE:
                st.caption(blocked)

            if "claude_report" in st.session_state:
                candidate, reply = st.session_state["claude_report"]
                st.subheader("Generated report")
                st.caption(f"Candidate {candidate}")
                st.markdown(reply.text)
                st.caption(f"Written by {reply.model}. Check every claim against the literature before using it.")


# ============================================================
# Descriptor Screening (DigMethpy Fig. 4B analogue)
# ============================================================

elif page == "Descriptor Screening":
    ui.page_header("Element map", "Modelled")
    st.markdown(
        "Every additive element is placed on a map of two elemental descriptors and tagged with what "
        "the literature database reports for it. Untested elements are ranked by how close they are, "
        "in standardized descriptor space, to elements that already gave the target product. "
        "(Map after Cheng et al. 2026, see About and references.)"
    )

    if not (descriptor_file.exists() and step3_file.exists()):
        st.warning("Descriptor or database file not found.")
    else:
        desc = load_excel_sheet(descriptor_file, "Element_descriptors")
        analyzed = load_excel_sheet(step3_file, "Analyzed_database")
        numeric = [c for c in scr_mod.NUMERIC_DESCRIPTORS if c in desc.columns]
        label = lambda c: scr_mod.NUMERIC_DESCRIPTORS.get(c, c)  # noqa: E731

        classes = sorted(analyzed["Product_class"].dropna().astype(str).unique())
        c1, c2, c3 = st.columns(3)
        with c1:
            target = st.selectbox(
                "Target product", classes, index=classes.index("solid carbon") if "solid carbon" in classes else 0
            )
        with c2:
            x = st.selectbox("x descriptor", numeric, index=numeric.index("Electronegativity_Pauling"), format_func=label)
        with c3:
            y_default = "Standard_reduction_potential_V_vs_SHE"
            y = st.selectbox("y descriptor", numeric, index=numeric.index(y_default), format_func=label)

        evidence = scr_mod.element_evidence(analyzed, desc, target)
        map_df = scr_mod.screening_map_data(desc, evidence, x, y)
        st.altair_chart(plot_mod.screening_map(map_df, label(x), label(y), ui.chart_ink()), width="stretch")
        n_untested = int((map_df["Status"] == scr_mod.STATUS_UNTESTED).sum())
        st.caption(f"{len(map_df)} elements plotted · {n_untested} not yet tested for '{target}'")
        dropped = sorted(set(desc["Element"]) - set(map_df["Element"]))
        if dropped:
            st.caption(f"Not plotted (descriptor value missing): {', '.join(dropped)}")

        st.subheader("Similarity ranking of untested elements")
        features = st.multiselect(
            "Descriptors used for similarity",
            numeric,
            default=[f for f in ["Electronegativity_Pauling", "Standard_reduction_potential_V_vs_SHE", "Atomic_radius_pm", "Melting_point_C"] if f in numeric],
            format_func=label,
        )
        if len(features) < 2:
            st.info("Pick at least two descriptors.")
        else:
            ranking, excluded = scr_mod.similarity_ranking(desc, evidence, features)
            if ranking.empty:
                st.info(f"No ranking: need at least one element with '{target}' reported and one untested element.")
            else:
                st.dataframe(
                    ranking,
                    width="stretch",
                    column_config={
                        "Descriptor distance": st.column_config.NumberColumn(format="%.2f"),
                        "Similarity score": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                st.caption(
                    "Heuristic: Euclidean distance in z-scored descriptor space to the nearest element with the "
                    "target product reported. It suggests what to test next; it is not a prediction of activity."
                    + (f" Excluded for missing values: {', '.join(excluded)}." if excluded else "")
                )

        with st.expander("Literature status per element"):
            st.dataframe(evidence, width="stretch", hide_index=True)


# ============================================================
# Computational Data (DFT / AIMD) and descriptor regression
# ============================================================

elif page == "Computational Data":
    ui.page_header("Computed properties", "Computed")
    st.markdown(
        "First-principles descriptors (Bader charge, adsorption energies, barriers, diffusivity) for solute "
        "atoms in liquid metals. Records are stored one property per row in `data/computational_data.csv`."
    )
    st.info(
        "Platform records are public data only: 0 K bulk formation energies (oxides, carbides, Ga–X compounds) "
        "from the [Open Quantum Materials Database](https://oqmd.org) (CC BY 4.0; Saal et al., JOM 2013; "
        "Kirklin et al., npj Comput. Mater. 2015). They describe bulk solids, not the molten alloy, and serve "
        "as thermodynamic descriptors until liquid-phase DFT/AIMD data are published."
    )

    stored, stored_issues = comp_mod.validate_computational(comp_mod.load_computational(computational_file))

    with st.expander("Add your own data for this session (not saved on the server)"):
        st.download_button(
            "Download CSV template",
            comp_mod.template_csv(),
            file_name="DigCO2_LM_computational_template.csv",
            mime="text/csv",
        )
        st.markdown(
            "Suggested `Property` names:\n"
            + "\n".join(f"- `{k}` — {v}" for k, v in comp_mod.SUGGESTED_PROPERTIES.items())
        )
        upload = st.file_uploader("Upload a CSV in the template format", type=["csv"], key="comp_upload")

    session_df = pd.DataFrame(columns=comp_mod.COMP_COLUMNS)
    if upload is not None:
        try:
            session_df, up_issues = comp_mod.validate_computational(pd.read_csv(upload))
            for msg in up_issues:
                st.warning(msg)
            st.success(f"Loaded {len(session_df)} uploaded record(s) for this session.")
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")

    comp = pd.concat(
        [stored.assign(Origin="Platform"), session_df.assign(Origin="Uploaded (session only)")], ignore_index=True
    )
    for msg in stored_issues:
        st.warning(msg)

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(comp))
    c2.metric("Solutes", comp["Solute"].nunique() if not comp.empty else 0)
    c3.metric("Properties", comp["Property"].nunique() if not comp.empty else 0)

    if comp.empty:
        st.info(
            "No computational records yet. Download the template, fill in DFT/AIMD results, and upload it "
            "above to explore them, or add reviewed rows to data/computational_data.csv to publish them."
        )
    else:
        safe_display(comp, height=320)

        st.subheader("Descriptor regression")
        st.caption(
            "Predicts a computed solute property from elemental features (approach after Cheng et al. 2026, see About and references). "
            "Leave-one-out CV over solutes, compared with predicting the mean."
        )
        desc = load_excel_sheet(descriptor_file, "Element_descriptors")
        numeric = [c for c in scr_mod.NUMERIC_DESCRIPTORS if c in desc.columns]
        props = sorted(comp["Property"].unique())
        c1, c2 = st.columns([1, 2])
        with c1:
            default_prop = "Oxide_formation_energy_per_O"
            prop = st.selectbox("Target property", props, index=props.index(default_prop) if default_prop in props else 0)
            model_name = st.radio("Model", ["Ridge", "RandomForest"], horizontal=True)
        with c2:
            features = st.multiselect(
                "Elemental features",
                numeric,
                default=[f for f in ["Electronegativity_Pauling", "Atomic_radius_pm", "Standard_reduction_potential_V_vs_SHE"] if f in numeric],
                format_func=lambda c: scr_mod.NUMERIC_DESCRIPTORS.get(c, c),
            )

        table, unknown = comp_mod.regression_table(comp, desc, prop, features) if features else (pd.DataFrame(), [])
        if unknown:
            st.caption(f"Solutes without complete elemental features (left out): {', '.join(unknown)}")
        min_n = max(5, len(features) + 2)
        if not features:
            st.info("Pick at least one elemental feature.")
        elif len(table) < min_n:
            st.info(
                f"'{prop}' has {len(table)} solute(s) with complete features; at least {min_n} are needed "
                f"for a meaningful leave-one-out fit with {len(features)} feature(s)."
            )
        else:
            result, m = comp_mod.loocv_regression(table, features, model_name)
            unit = comp.loc[comp["Property"] == prop, "Unit"].dropna().astype(str)
            unit = unit.iloc[0] if not unit.empty else ""
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("RMSE", f"{m['model']['RMSE']:.3f}", f"{m['model']['RMSE'] - m['baseline']['RMSE']:+.3f} vs mean", delta_color="inverse")
            mc2.metric("MAE", f"{m['model']['MAE']:.3f}", f"{m['model']['MAE'] - m['baseline']['MAE']:+.3f} vs mean", delta_color="inverse")
            mc3.metric("R²", f"{m['model']['R2']:.2f}")
            if m["model"]["RMSE"] >= m["baseline"]["RMSE"]:
                st.warning("The model does not beat predicting the mean. More solutes or different features are needed.")
            st.markdown(f"**{prop}: computed vs LOOCV prediction** ({model_name}, {len(result)} solutes)")
            x_title = f"Computed {prop} ({unit})" if unit else f"Computed {prop}"
            st.altair_chart(plot_mod.parity_plot(result, x_title, ui.chart_ink()), width="stretch")
            st.dataframe(result, width="stretch", hide_index=True)


# ============================================================
# Literature Agent (retrieval-augmented)
# ============================================================

elif page == "Literature Agent":
    ui.page_header("Ask the records", "Reported")
    st.markdown(
        "Ask about mechanisms, conditions or products. The page retrieves the most relevant curated records "
        "and, when Claude is connected, writes an answer that may only use those records and cites them as [R#]."
    )

    if not descriptor_file.exists():
        st.warning("Database file not found.")
    else:
        retriever = get_retriever()
        k = st.slider("Records to retrieve", 2, 8, 4)

        history = st.session_state.setdefault("lit_chat", [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("e.g. Which additives help CO2-to-solid-carbon at low temperature, and why?")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            hits = retriever.search(question, k=k)

            with st.chat_message("assistant"):
                if hits.empty:
                    answer = "No record in the database matches this question."
                    st.markdown(answer)
                else:
                    with st.expander(f"Retrieved evidence ({len(hits)} records)", expanded=False):
                        for h in hits.itertuples():
                            link = f" · [source]({h.Link})" if h.Link else ""
                            st.markdown(f"**[R{h.Record_ID}] {h.Title}** ({h.Year or 'n.d.'}) · score {h.Score:.2f}{link}")
                            st.caption(h.Text.replace("\n", " · "))

                    blocked = claude_block_reason()
                    if blocked:
                        answer = f"{blocked} Here are the most relevant records: " + ", ".join(
                            f"[R{h.Record_ID}] {h.Title}" for h in hits.itertuples()
                        )
                    else:
                        with st.spinner("Writing a grounded answer..."):
                            reply = call_claude(rag_mod.build_prompt(question, hits, history))
                        answer = reply.text if reply.ok else f"{reply.text} The retrieved records are listed above."
                    st.markdown(answer)

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

        if history and st.button("Clear conversation"):
            st.session_state["lit_chat"] = []
            st.rerun()


# ============================================================
# Experimental Feedback (closed loop)
# ============================================================

elif page == "Experimental Feedback":
    ui.page_header("Submit a result", "Tested")
    st.markdown(
        "Report an experimental result. The platform compares it with the current model "
        "(**agreement** or **discrepancy**; neither proves the result), saves it as "
        "*pending*, and a maintainer moves reviewed entries into the database."
    )

    analyzed = load_excel_sheet(step3_file, "Analyzed_database") if step3_file.exists() else pd.DataFrame()
    known_classes = sorted(analyzed["Product_class"].dropna().astype(str).unique()) if not analyzed.empty else []
    trained_reactions = training_categories("Reaction_type_for_ML")
    desc = load_excel_sheet(descriptor_file, "Element_descriptors") if descriptor_file.exists() else pd.DataFrame()
    elements = sorted(desc["Element"].dropna().astype(str).unique()) if not desc.empty else []

    def opt_num(label, **kw):
        v = st.number_input(label, value=None, placeholder="not measured", **kw)
        return None if v is None else float(v)

    with st.form("feedback_form", clear_on_submit=False):
        st.markdown("**System and conditions**")
        c1, c2, c3 = st.columns(3)
        with c1:
            lm_comp = st.text_input("Liquid-metal composition *", placeholder="e.g. Ga-Zn (8 at% Zn)")
            base_lm = st.text_input("Base liquid metal", placeholder="e.g. Ga, EGaIn, Galinstan")
            additive = st.selectbox("Additive element", ["(none / other)"] + elements)
        with c2:
            reaction = st.selectbox("Reaction type *", trained_reactions or ["thermal", "electrochemical"])
            temp = opt_num("Temperature (°C)", step=10.0)
            pressure = st.text_input("Pressure", placeholder="e.g. 1 atm")
        with c3:
            voltage = opt_num("Voltage (V)", step=0.1)
            current = opt_num("Current density (mA/cm²)", step=0.1)
            rtime = opt_num("Reaction time (h)", step=0.5)

        st.markdown("**Outcome**")
        c1, c2, c3 = st.columns(3)
        with c1:
            main_product = st.text_input("Main product(s) *", placeholder="e.g. amorphous carbon flakes")
            product_class = st.selectbox("Product class *", known_classes + ["other (describe in main product)"])
        with c2:
            conversion = opt_num("CO₂ conversion (%)", min_value=0.0, max_value=100.0, step=1.0)
            fe_total = opt_num("Faradaic efficiency total (%)", min_value=0.0, max_value=100.0, step=1.0)
        with c3:
            carbon_rate = opt_num("Carbon rate (µmol/h)", min_value=0.0, step=1.0)
            charac = st.text_input("Characterization", placeholder="e.g. Raman; XPS; SEM")

        st.markdown("**Provenance**")
        c1, c2, c3 = st.columns(3)
        with c1:
            data_type = st.selectbox("Data type", ["Own experiment (unpublished)", "Own experiment (published)", "Literature not yet in database"])
        with c2:
            source = st.text_input("DOI / URL / lab notebook ID")
        with c3:
            submitter = st.text_input("Name or group (optional)")

        submitted = st.form_submit_button("Check against model and submit")

    if submitted:
        entry = {
            "Submitter": submitter.strip() or None,
            "Data_type": data_type,
            "LM_composition": lm_comp.strip(),
            "Base_LM": base_lm.strip() or None,
            "Additive_metal_clean": None if additive.startswith("(") else additive,
            "Reaction_type_clean": reaction,
            "Temp_representative_C": temp,
            "Pressure_clean": pressure.strip() or None,
            "Voltage_V": voltage,
            "Current_density_mA_cm2": current,
            "Reaction_time_h_value": rtime,
            "Main_product": main_product.strip(),
            "Product_class": product_class,
            "CO2_conversion_representative_%": conversion,
            "FE_total_%": fe_total,
            "Carbon_rate_umol_h": carbon_rate,
            "Characterization_methods": charac.strip() or None,
            "Source": source.strip() or None,
            "Review_status": "pending",
        }
        problems = fb_mod.validate_submission(entry)
        if problems:
            for p in problems:
                st.error(p)
        else:
            if model_file.exists() and not desc.empty:
                lm_known = set(training_categories("Liquid_metal_clean_for_ML"))
                condition = make_condition_for_model(
                    reaction,
                    lm_comp if lm_comp in lm_known else "Not reported",
                    entry["Additive_metal_clean"] or "Not reported",
                    temp or 0.0,
                    voltage or 0.0,
                    current or 0.0,
                    desc,
                )
                pred, prob_df = predict_condition(condition)
                prob = float(prob_df.loc[prob_df["Product_class"] == pred, "Probability"].iloc[0]) if prob_df is not None else None
                entry["Model_predicted_class"] = pred
                entry["Model_probability"] = prob
                entry["Model_agreement"] = "agreement" if pred == product_class else "discrepancy"

                if entry["Model_agreement"] == "agreement":
                    st.success(f"Agreement: the model also predicts '{pred}'" + (f" (p = {prob:.2f})." if prob is not None else "."))
                else:
                    st.warning(
                        f"Discrepancy: the model predicts '{pred}'" + (f" (p = {prob:.2f})" if prob is not None else "")
                        + f" but you observed '{product_class}'. Discrepancies are the most valuable feedback for retraining."
                    )
                st.caption("The classifier does not beat the majority-class baseline: agreement does not confirm a result.")

            fb_list = st.session_state.setdefault("feedback_submissions", [])
            entry["Feedback_ID"] = fb_mod.new_feedback_id(len(fb_list))
            entry["Submitted_at_UTC"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
            row = fb_mod.as_row(entry)
            fb_list.append(row)
            ok, msg = fb_mod.append_pending(pending_feedback_file, row)
            (st.info if ok else st.warning)(msg)

    subs = st.session_state.get("feedback_submissions", [])
    if subs:
        st.subheader("Your submissions in this session")
        sub_df = pd.DataFrame(subs, columns=fb_mod.FEEDBACK_COLUMNS)
        safe_display(sub_df, height=220)
        st.download_button(
            "Download my submissions (CSV)",
            sub_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="DigCO2_LM_feedback_submissions.csv",
            mime="text/csv",
        )
        st.caption(
            "Email this CSV to the platform maintainers, or open a pull request adding it to "
            "data/feedback/. On the hosted app the server copy is temporary."
        )

    curated = fb_mod.load_feedback(curated_feedback_file)
    st.subheader("Curated feedback in the database")
    if curated.empty:
        st.caption("No curated feedback yet.")
    else:
        agree = (curated["Model_agreement"] == "agreement").mean()
        st.caption(f"{len(curated)} record(s) · model agreement {agree:.0%}")
        safe_display(curated, height=260)

    if ADMIN_MODE:
        st.subheader("Admin: pending feedback on this server")
        pending = fb_mod.load_feedback(pending_feedback_file)
        st.caption(f"{len(pending)} pending record(s). Review, then append accepted rows to data/feedback/curated_feedback.csv.")
        if not pending.empty:
            safe_display(pending, height=260)
            st.download_button(
                "Download pending feedback",
                pending.to_csv(index=False, encoding="utf-8-sig"),
                file_name="pending_feedback.csv",
                mime="text/csv",
            )


# ============================================================
# File Status
# ============================================================

elif page == "File Status":
    ui.page_header("File status")

    files = {
        "app.py": base_dir / "app.py",
        "requirements.txt": base_dir / "requirements.txt",
        "Dockerfile": base_dir / "Dockerfile",
        "Step3_LM_CO2_database_analysis.xlsx": step3_file,
        "Step4_ML_ready_dataset.xlsx": step4_file,
        "Step6_agent_recommendation_results.xlsx": step6_file,
        "Step7b_Gemini_LLM_agent_report.xlsx": step7b_file,
        "Final_LM_CO2_platform_database_with_descriptors.xlsx": descriptor_file,
        "product_classifier_random_forest.pkl": model_file,
    }

    status_df = pd.DataFrame(
        [
            {"File": name, "Status": "Found" if path.exists() else "Missing", "Path": str(path)}
            for name, path in files.items()
        ]
    )

    safe_display(status_df, height=400)

    st.subheader("Base directory")
    st.code(str(base_dir))

    st.subheader("Claude API")
    problem = CLAUDE.problem()
    if problem:
        st.warning(problem)
    else:
        st.success(f"Connected: {CLAUDE.model}, {claude_daily_limit().remaining()} calls left today.")


render_footer()