import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from digco2 import computational as comp_mod
from digco2 import feedback as fb_mod
from digco2 import literature_rag as rag_mod
from digco2 import screening as scr_mod
from digco2 import plots as plot_mod

try:
    from google import genai
except Exception:
    genai = None


# ============================================================
# DigCO2-LM Streamlit App
# DigMethpy-style platform UI
# ============================================================

st.set_page_config(
    page_title="DigCO2-LM",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

base_dir = Path(__file__).parent

APP_VERSION = "0.3.0"

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
computational_file = data_dir / "computational_data.csv"
pending_feedback_file = data_dir / "feedback" / "pending_feedback.csv"
curated_feedback_file = data_dir / "feedback" / "curated_feedback.csv"


# ============================================================
# CSS styling
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .hero {
        padding: 2.2rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f766e 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.20);
    }

    .hero h1 {
        font-size: 3.0rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        max-width: 900px;
    }

    .section-card {
        background: white;
        padding: 1.3rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .small-card {
        background: white;
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
        min-height: 125px;
    }

    .module-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
    }

    .muted {
        color: #64748b;
        font-size: 0.92rem;
    }

    .pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: #e0f2fe;
        color: #075985;
        font-weight: 600;
        font-size: 0.82rem;
        margin-right: 0.3rem;
        margin-bottom: 0.25rem;
    }

    .warning-box {
        padding: 1rem;
        border-radius: 16px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
    }

    .success-box {
        padding: 1rem;
        border-radius: 16px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }

    .workflow-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
        min-height: 105px;
    }

    .arrow {
        text-align: center;
        font-size: 2rem;
        color: #1e40af;
        padding-top: 1.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helper functions
# ============================================================

@st.cache_data
def load_excel_sheet(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


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
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-2.5-flash")
LIVE_GEMINI_LIMIT_PER_SESSION = 5
OTHER_LM_LABEL = "Other (not in training data)"


@st.cache_data
def training_categories(column):
    """Category values the classifier actually saw during training (Step4 ML_ready_dataset)."""
    if not step4_file.exists():
        return []
    df = load_excel_sheet(step4_file, "ML_ready_dataset")
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).str.strip().unique())


def render_model_performance():
    """Compare the classifier with a majority-class baseline, using Step4 outputs."""
    if not step4_file.exists():
        return
    xls = pd.ExcelFile(step4_file, engine="openpyxl")

    with st.expander("Model performance vs. majority-class baseline", expanded=True):
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
                rows.append({"Model": r["Model"], "Accuracy": r["Accuracy_or_CV_score"], "Note": r.get("Comment", "")})
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
        f"DigCO2-LM v{APP_VERSION} · For research and educational purposes only. "
        "Predictions are proof-of-concept screening suggestions from a small literature-derived dataset "
        "and are not validated experimental results."
    )
    st.caption(
        "Platform design inspired by DigMethpy: Cheng, Z. et al. AI Agent 2026, 2, 8. "
        "[DOI: 10.20517/aiagent.2026.11](https://dx.doi.org/10.20517/aiagent.2026.11)"
    )


def get_gemini_client():
    api_key = get_setting("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY was not found in Streamlit secrets or environment variables."
    if genai is None:
        return None, "google-genai package is not installed or could not be imported."
    try:
        return genai.Client(api_key=api_key), None
    except Exception as e:
        return None, str(e)


def call_gemini(prompt, model_name=GEMINI_MODEL):
    client, error = get_gemini_client()
    if error:
        return f"Gemini client error: {error}"
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Gemini API call failed: {e}"


def normalize_descriptor_row(desc_df, additive):
    row = desc_df[desc_df["Element"].astype(str).str.strip() == additive]
    if row.empty:
        return {}

    r = row.iloc[0]

    def g(col, default="Not reported"):
        if col in r.index and pd.notna(r[col]):
            return r[col]
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
def get_retriever():
    core = load_excel_sheet(descriptor_file, "Core_database")
    mech = load_excel_sheet(descriptor_file, "Mechanism_notes")
    return rag_mod.Retriever(rag_mod.build_corpus(core, mech))


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


def build_gemini_prompt_from_candidate(row):
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
# Sidebar navigation
# ============================================================

st.sidebar.markdown("## ⚗️ DigCO2-LM")
st.sidebar.caption("Digital CO₂ Conversion Platform")

pages = [
    "Home",
    "Platform Workflow",
    "Database",
    "Descriptor Explorer",
    "Descriptor Screening",
    "Computational Data",
    "ML Prediction",
    "Recommendation Agent",
    "Literature Agent",
    "Gemini Agent Report",
    "Live Gemini Generator",
    "Experimental Feedback",
]
if ADMIN_MODE:
    # Server paths and API-key status are only shown when DIGCO2_ADMIN = "1" is set in secrets.
    pages.append("File Status")

page = st.sidebar.radio("Platform modules", pages)

st.sidebar.divider()
st.sidebar.markdown("### System status")
st.sidebar.write(f"Database: {file_status(step3_file)}")
st.sidebar.write(f"Descriptors: {file_status(descriptor_file)}")
st.sidebar.write(f"ML model: {file_status(model_file)}")
st.sidebar.write(f"Agent results: {file_status(step6_file)}")
st.sidebar.write(f"LLM report: {file_status(step7b_file)}")
st.sidebar.caption(f"Version {APP_VERSION}")


# ============================================================
# Home page
# ============================================================

if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>DigCO2-LM</h1>
            <p>
            An AI-enabled digital catalysis platform for liquid-metal-mediated CO₂ conversion.
            The platform integrates literature-mined databases, elemental descriptors,
            machine-learning product classification, and Gemini-powered recommendation reports.
            </p>
            <span class="pill">Liquid metals</span>
            <span class="pill">CO₂ conversion</span>
            <span class="pill">Product classification</span>
            <span class="pill">LLM agent</span>
            <span class="pill">Experimental feedback</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if step3_file.exists():
        df = load_excel_sheet(step3_file, "Analyzed_database")
        n_records = len(df)
        n_products = df["Product_class"].nunique() if "Product_class" in df.columns else "N/A"
        n_reactions = df["Reaction_type_clean"].nunique() if "Reaction_type_clean" in df.columns else "N/A"
        n_additives = df["Additive_metal_clean"].nunique() if "Additive_metal_clean" in df.columns else "N/A"

        comp_df, _ = comp_mod.validate_computational(comp_mod.load_computational(computational_file))
        n_curated = len(fb_mod.load_feedback(curated_feedback_file))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Literature records", n_records)
        c2.metric("Computational records", len(comp_df))
        c3.metric("Curated feedback", n_curated)
        c4.metric("Product classes", n_products)
        st.caption(f"{n_reactions} reaction modes · {n_additives} additive-metal entries")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            if "Product_class" in df.columns:
                st.subheader("Product landscape")
                st.bar_chart(
                    df["Product_class"].fillna("Not reported").value_counts(),
                    horizontal=True,
                    x_label="",
                    y_label="Records",
                )

        with c2:
            if "Reaction_type_clean" in df.columns:
                st.subheader("Reaction-mode landscape")
                st.bar_chart(
                    df["Reaction_type_clean"].fillna("Not reported").value_counts(),
                    horizontal=True,
                    x_label="",
                    y_label="Records",
                )

    else:
        st.warning("Database file is missing. Please upload or include Step3_LM_CO2_database_analysis.xlsx.")

    st.markdown("## Platform modules")
    modules = [
        ("Database", "Curated literature-mined LM–CO₂ conversion records with source links."),
        ("Descriptor Screening", "Place untested additive elements against known solid-carbon formers."),
        ("Computational Data", "DFT / AIMD descriptors and leave-one-out descriptor regression."),
        ("ML Classifier", "Preliminary product-class model, shown against a majority baseline."),
        ("Literature Agent", "Ask questions answered only from the curated records, with citations."),
        ("Experimental Feedback", "Submit results, compare with the model, and feed the database."),
    ]
    for start in range(0, len(modules), 3):
        cols = st.columns(3)
        for col, (name, text) in zip(cols, modules[start : start + 3]):
            with col:
                st.markdown(
                    f'<div class="small-card"><div class="module-title">{name}</div><p class="muted">{text}</p></div>',
                    unsafe_allow_html=True,
                )
        st.write("")


# ============================================================
# Platform Workflow
# ============================================================

elif page == "Platform Workflow":
    st.header("🧭 Platform Workflow")

    st.markdown(
        """
        <div class="section-card">
        <b>DigCO2-LM workflow</b> connects literature mining, database construction,
        descriptor integration, machine learning, recommendation agents, and experimental feedback.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 0.25, 1, 0.25, 1, 0.25, 1])
    with cols[0]:
        st.markdown('<div class="workflow-box"><b>1. Literature</b><br>PDFs and papers on LM–CO₂ conversion</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="workflow-box"><b>2. Database</b><br>Reaction conditions, products, performance</div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown('<div class="workflow-box"><b>3. Descriptors</b><br>Oxygen affinity, reductant strength, element properties</div>', unsafe_allow_html=True)
    with cols[5]:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with cols[6]:
        st.markdown('<div class="workflow-box"><b>4. ML</b><br>Product-class prediction</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns([1, 0.25, 1, 0.25, 1])
    with cols[0]:
        st.markdown('<div class="workflow-box"><b>5. Agent</b><br>Candidate generation and scoring</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="workflow-box"><b>6. Gemini</b><br>Experimental planning and interpretation</div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown('<div class="workflow-box"><b>7. Feedback</b><br>Validation data updates the database</div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("Research positioning")
    st.markdown(
        """
        - **Data-centric:** structured literature data are treated as reusable catalyst-design assets.
        - **Descriptor-aware:** elemental properties are connected to product trends.
        - **Model-assisted:** ML is used as a proof-of-concept classifier, not a definitive predictor.
        - **Agentic:** the recommendation agent combines ML scores, descriptor rules, and LLM reasoning.
        - **Closed-loop:** future experimental results can be added back to improve the database and model.
        """
    )

    st.subheader("Implementation status")
    st.dataframe(
        pd.DataFrame(
            [
                ("Literature database", "Available", "18 curated records with source links"),
                ("Element descriptors + screening map", "Available", "Similarity heuristic, not an activity model"),
                ("Computational data (DFT/AIMD)", "Schema ready", "Records are added after review; session upload available"),
                ("Descriptor regression", "Available when data exist", "Needs ≥5 solutes per property"),
                ("Product classifier", "Preliminary", "Does not beat the majority-class baseline yet"),
                ("Literature agent (RAG)", "Available", "Answers only from curated records; Gemini optional"),
                ("Experimental feedback", "Available", "Pending → maintainer review → curated"),
                ("Automated literature mining", "Planned", "PDF parsing + LLM extraction with QC curation"),
                ("ML interatomic potentials", "Planned", "For large-scale molten-surface sampling"),
            ],
            columns=["Module", "Status", "Note"],
        ),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# Database
# ============================================================

elif page == "Database":
    st.header("🧾 Liquid-metal CO₂ Conversion Database")

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

        st.write(f"Filtered records: {len(filtered)} / {len(df)}")

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
            "Record_ID",
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


# ============================================================
# Descriptor Explorer
# ============================================================

elif page == "Descriptor Explorer":
    st.header("🧬 Element Descriptor Explorer")

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
    st.header("🔮 Product-class Prediction")

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

        st.markdown(
            """
            <div class="warning-box">
            This model is trained on a small literature-derived dataset.
            Predictions should be interpreted as proof-of-concept screening suggestions.
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            voltage_v = st.number_input("Voltage (V)", value=0.0, step=0.1)
            current_density = st.number_input("Current density (mA/cm²)", value=0.0, step=0.1)

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
    st.header("🧠 Recommendation Agent")

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
    st.header("📝 Gemini Agent Report")

    if not step7b_file.exists():
        st.warning("Step7b_Gemini_LLM_agent_report.xlsx not found.")
    else:
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
# Live Gemini Generator
# ============================================================

elif page == "Live Gemini Generator":
    st.header("✨ Live Gemini Generator")

    st.markdown(
        """
        <div class="warning-box">
        This module calls the Gemini API in real time.
        If the app is publicly shared, API usage may increase.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

            prompt = build_gemini_prompt_from_candidate(row)
            st.caption(f"Model: {GEMINI_MODEL} (set GEMINI_MODEL in secrets to change)")

            with st.expander("Prompt preview"):
                st.code(prompt)

            used = st.session_state.get("live_gemini_calls", 0)
            remaining = LIVE_GEMINI_LIMIT_PER_SESSION - used
            st.caption(
                f"Live Gemini calls left in this session: {remaining} / {LIVE_GEMINI_LIMIT_PER_SESSION} "
                "(shared with the Literature Agent)"
            )

            if st.button("Generate live Gemini report", disabled=remaining <= 0):
                st.session_state["live_gemini_calls"] = used + 1
                with st.spinner("Calling Gemini API..."):
                    report = call_gemini(prompt)
                st.subheader("Generated report")
                st.markdown(report)


# ============================================================
# Descriptor Screening (DigMethpy Fig. 4B analogue)
# ============================================================

elif page == "Descriptor Screening":
    st.header("🗺️ Descriptor Screening")
    st.markdown(
        "Every additive element is placed on a map of two elemental descriptors and tagged with what "
        "the literature database reports for it. Untested elements are ranked by how close they are, "
        "in standardized descriptor space, to elements that already gave the target product."
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
        st.altair_chart(plot_mod.screening_map(map_df, label(x), label(y)), width="stretch")
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
    st.header("🧮 Computational Data")
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
            "Predicts a computed solute property from elemental features (DigMethpy Fig. 4A approach). "
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
            st.altair_chart(plot_mod.parity_plot(result, x_title), width="stretch")
            st.dataframe(result, width="stretch", hide_index=True)


# ============================================================
# Literature Agent (retrieval-augmented)
# ============================================================

elif page == "Literature Agent":
    st.header("📚 Literature Agent")
    st.markdown(
        "Ask about mechanisms, conditions or products. The agent retrieves the most relevant curated records "
        "and, if Gemini is configured, writes an answer that may only use those records and cites them as [R#]."
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

                    used = st.session_state.get("live_gemini_calls", 0)
                    client, err = get_gemini_client()
                    if err:
                        answer = (
                            "Gemini is not configured, so here are the most relevant records: "
                            + ", ".join(f"[R{h.Record_ID}] {h.Title}" for h in hits.itertuples())
                        )
                    elif used >= LIVE_GEMINI_LIMIT_PER_SESSION:
                        answer = "The live Gemini limit for this session has been reached. The retrieved records are listed above."
                    else:
                        st.session_state["live_gemini_calls"] = used + 1
                        with st.spinner("Writing a grounded answer..."):
                            answer = call_gemini(rag_mod.build_prompt(question, hits, history))
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
    st.header("🔁 Experimental Feedback")
    st.markdown(
        "Report an experimental result. The platform compares it with the current model "
        "(**agreement** validates the prediction, a **discrepancy** is flagged for analysis), saves it as "
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
    st.header("📁 File Status")

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

    st.subheader("Gemini API environment")
    if get_setting("GEMINI_API_KEY"):
        st.success("GEMINI_API_KEY is available.")
    else:
        st.warning("GEMINI_API_KEY is not available.")


render_footer()