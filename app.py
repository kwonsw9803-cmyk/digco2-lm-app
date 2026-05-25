import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

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

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

step3_file = base_dir / "Step3_LM_CO2_database_analysis.xlsx"
step4_file = base_dir / "Step4_ML_ready_dataset.xlsx"
step6_file = base_dir / "Step6_agent_recommendation_results.xlsx"
step7b_file = base_dir / "Step7b_Gemini_LLM_agent_report.xlsx"
descriptor_file = base_dir / "Final_LM_CO2_platform_database_with_descriptors.xlsx"
model_file = base_dir / "product_classifier_random_forest.pkl"


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


def safe_display(df, height=500):
    st.dataframe(df, use_container_width=True, height=height)


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY environment variable was not found."
    if genai is None:
        return None, "google-genai package is not installed or could not be imported."
    try:
        return genai.Client(api_key=api_key), None
    except Exception as e:
        return None, str(e)


def call_gemini(prompt, model_name="gemini-2.5-flash"):
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

page = st.sidebar.radio(
    "Platform modules",
    [
        "Home",
        "Platform Workflow",
        "Database",
        "Descriptor Explorer",
        "ML Prediction",
        "Recommendation Agent",
        "Gemini Agent Report",
        "File Status",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### System status")
st.sidebar.write(f"Database: {file_status(step3_file)}")
st.sidebar.write(f"Descriptors: {file_status(descriptor_file)}")
st.sidebar.write(f"ML model: {file_status(model_file)}")
st.sidebar.write(f"Agent results: {file_status(step6_file)}")
st.sidebar.write(f"LLM report: {file_status(step7b_file)}")


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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Database records", n_records)
        c2.metric("Product classes", n_products)
        c3.metric("Reaction modes", n_reactions)
        c4.metric("Additive metals", n_additives)

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Product landscape")
            if "Product_class" in df.columns:
                st.bar_chart(df["Product_class"].fillna("Not reported").value_counts())
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Reaction-mode landscape")
            if "Reaction_type_clean" in df.columns:
                st.bar_chart(df["Reaction_type_clean"].fillna("Not reported").value_counts())
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Database file is missing. Please upload or include Step3_LM_CO2_database_analysis.xlsx.")

    st.markdown("## Platform modules")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            '<div class="small-card"><div class="module-title">Database</div><p class="muted">Curated literature-mined LM–CO₂ conversion records.</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="small-card"><div class="module-title">Descriptors</div><p class="muted">Elemental properties and chemical role descriptors.</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            '<div class="small-card"><div class="module-title">ML Classifier</div><p class="muted">Preliminary product-class prediction model.</p></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            '<div class="small-card"><div class="module-title">LLM Agent</div><p class="muted">Gemini-assisted interpretation and experiment planning.</p></div>',
            unsafe_allow_html=True,
        )


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


# ============================================================
# Database
# ============================================================

elif page == "Database":
    st.header("🧾 Liquid-metal CO₂ Conversion Database")

    if not step3_file.exists():
        st.warning("Step3 database file not found.")
    else:
        df = load_excel_sheet(step3_file, "Analyzed_database")
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

        key_cols = [
            "Record_ID",
            "Paper_title",
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

        safe_display(filtered[key_cols], height=600)

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

        with c1:
            if "Oxygen_affinity_class" in desc.columns:
                st.markdown("**Oxygen affinity class**")
                st.bar_chart(desc["Oxygen_affinity_class"].fillna("Not reported").value_counts())

        with c2:
            if "Reductant_strength_class" in desc.columns:
                st.markdown("**Reductant strength class**")
                st.bar_chart(desc["Reductant_strength_class"].fillna("Not reported").value_counts())


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

        st.divider()

        c1, c2, c3 = st.columns(3)
        with c1:
            reaction_type = st.selectbox(
                "Reaction type",
                ["thermal", "electrochemical", "mechanical", "plasma", "sonochemical"],
            )
            liquid_metal = st.selectbox(
                "Liquid metal",
                ["Ga", "EGaIn", "GaInSn", "GaIn", "Sn-Bi", "Ga-In-Mg", "Other"],
            )

        with c2:
            additive = st.selectbox(
                "Additive metal",
                sorted(desc["Element"].dropna().astype(str).unique()),
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

            pred_df = pd.DataFrame([condition])

            for col in training_columns:
                if col not in pred_df.columns:
                    if col in categorical_features:
                        pred_df[col] = "Not reported"
                    else:
                        pred_df[col] = 0

            X = pred_df[training_columns].copy()

            for col in categorical_features:
                X[col] = X[col].fillna("Not reported").astype(str)

            for col in numeric_features:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

            pred = model.predict(X)[0]

            st.success(f"Predicted product class: {pred}")

            if hasattr(model.named_steps["classifier"], "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = model.named_steps["classifier"].classes_
                prob_df = pd.DataFrame({"Product_class": classes, "Probability": proba})
                prob_df = prob_df.sort_values("Probability", ascending=False)
                st.subheader("Prediction probability")
                st.dataframe(prob_df, use_container_width=True)
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

        st.subheader("Top recommendations")
        safe_display(top_df[show_cols], height=500)

        st.download_button(
            "Download recommendation results",
            top_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="DigCO2_LM_top_recommendations.csv",
            mime="text/csv",
        )

        st.divider()

        st.subheader("Candidate detail")
        options = [
            f"{i+1}. {row.get('Liquid_metal_clean_for_ML', 'LM')}-{row.get('Additive_element_for_descriptor', 'Additive')} | {row.get('Reaction_type_for_ML', '')}"
            for i, row in top_df.iterrows()
        ]

        if options:
            selected = st.selectbox("Select candidate", options)
            row = top_df.iloc[options.index(selected)]

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
            model_name = st.text_input("Gemini model", value="gemini-2.5-flash")

            with st.expander("Prompt preview"):
                st.code(prompt)

            if st.button("Generate live Gemini report"):
                with st.spinner("Calling Gemini API..."):
                    report = call_gemini(prompt, model_name=model_name)
                st.subheader("Generated report")
                st.markdown(report)


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
    if os.environ.get("GEMINI_API_KEY"):
        st.success("GEMINI_API_KEY is available.")
    else:
        st.warning("GEMINI_API_KEY is not available.")