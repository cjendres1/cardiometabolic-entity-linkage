import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import duckdb
import random
import json
import os

# --- Splink 4.x Imports ---
from splink import Linker, DuckDBAPI, block_on
import splink.comparisons as cl

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiometabolic Entity Linker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Cardiometabolic Entity Linkage Dashboard")
st.markdown(
    "Probabilistic record linkage across **CDC NHANES (2009–2018)** cohorts using "
    "the **Fellegi-Sunter model** (`Splink` + `DuckDB`)."
)

# -----------------------------------------------------------------------------
# PIPELINE INITIALIZATION & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    """Loads NHANES data and injects synthetic EHR noise for demo purposes."""
    csv_path = "data/nhanes_2009_2018_cardiometabolic.csv"
    
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
    else:
        # Fallback mock data if CSV is not yet generated
        np.random.seed(42)
        n_rows = 1000
        df_raw = pd.DataFrame({
            "SEQN": range(100000, 100000 + n_rows),
            "age": np.random.randint(18, 80, size=n_rows),
            "gender": np.random.choice(["Male", "Female"], size=n_rows),
            "hba1c": np.round(np.random.normal(5.7, 1.2, size=n_rows), 1),
            "fasting_glucose": np.random.randint(70, 180, size=n_rows),
            "sys_bp": np.random.randint(110, 160, size=n_rows),
            "diabetes_status": np.random.choice(["Yes", "No", "Borderline"], size=n_rows),
            "cycle": np.random.choice(["2009-2010", "2011-2012", "2013-2014", "2015-2016", "2017-2018"], size=n_rows)
        })

    # Synthetic identity injection (First & Last Names)
    first_names = ["James", "John", "Robert", "Michael", "William", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

    np.random.seed(42)
    df_raw["unique_id"] = df_raw["SEQN"].astype(str)
    df_raw["first_name"] = np.random.choice(first_names, size=len(df_raw))
    df_raw["last_name"] = np.random.choice(last_names, size=len(df_raw))

    # Inject 10% duplicates with noise (typos / slight metric variances)
    sample_size = max(10, int(len(df_raw) * 0.10))
    duplicates = df_raw.sample(n=sample_size, random_state=42).copy()

    def inject_typo(name):
        if len(str(name)) > 3 and random.random() > 0.4:
            idx = random.randint(0, len(str(name)) - 1)
            return name[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + name[idx+1:]
        return name

    duplicates["unique_id"] = duplicates["unique_id"] + "_dup"
    duplicates["first_name"] = duplicates["first_name"].apply(inject_typo)
    duplicates["age"] = duplicates["age"] + np.random.choice([-1, 0, 1], size=len(duplicates))
    duplicates["hba1c"] = np.round(duplicates["hba1c"] + np.random.normal(0, 0.2, size=len(duplicates)), 1)

    df_combined = pd.concat([df_raw, duplicates], ignore_index=True)
    return df_combined

@st.cache_resource
def run_splink_pipeline(df):
    db_api = DuckDBAPI()

    settings = {
        "link_type": "dedupe_only",
        "blocking_rules_to_generate_predictions": [
            block_on("first_name", "gender"),
            block_on("last_name", "gender"),
            block_on("age", "gender")
        ],
        "comparisons": [
            ctl.NameComparison("first_name", jaro_winkler_thresholds=[0.88, 0.94]),
            ctl.NameComparison("last_name", jaro_winkler_thresholds=[0.88, 0.94]),
            cl.ExactMatch("gender"),
            cl.NumericDifferenceAtThresholds("age", thresholds=[1, 3, 5]),
            cl.NumericDifferenceAtThresholds("hba1c", thresholds=[0.2, 0.5, 1.0]),
        ],
        "retain_matching_framework": True,
        "retain_intermediate_calculation_columns": True
    }

    linker = Linker(df, settings, db_api=db_api)
    linker.training.estimate_u_probability_two_random_records_match(max_pairs=50_000, seed=42)
    linker.training.estimate_parameters_using_expectation_maximization(block_on("first_name"))
    linker.training.estimate_parameters_using_expectation_maximization(block_on("last_name"))

    predictions = linker.inference.predict(match_weight_threshold=-5.0)
    df_preds = predictions.as_pandas_dataframe()

    return linker, df_preds

# Load Data and Train Linker
df_records = load_and_prepare_data()
linker, df_predictions = run_splink_pipeline(df_records)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Linkage Filters")

min_match_weight = float(df_predictions["match_weight"].min())
max_match_weight = float(df_predictions["match_weight"].max())

weight_threshold = st.sidebar.slider(
    "Minimum Match Weight Threshold",
    min_value=round(min_match_weight, 1),
    max_value=round(max_match_weight, 1),
    value=2.0,
    step=0.5,
    help="Match weights represent the log2 Bayes factor. Higher weights indicate stronger match probability."
)

filtered_preds = df_predictions[df_predictions["match_weight"] >= weight_threshold].copy()

# -----------------------------------------------------------------------------
# METRICS TOP BAR
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Input Records", f"{len(df_records):,}")
col2.metric("Evaluated Candidate Pairs", f"{len(df_predictions):,}")
col3.metric("Predicted High-Probability Links", f"{len(filtered_preds):,}")
col4.metric("Max Match Weight", f"{max_match_weight:.2f}")

st.markdown("---")

# -----------------------------------------------------------------------------
# TABBED INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Pairwise Linkage Explorer", "🌊 Match Waterfall Diagnostics", "📈 Model Parameters"])

# --- TAB 1: PAIRWISE EXPLORER ---
with tab1:
    st.subheader("Predicted Matches Table")
    st.markdown("Filter candidates based on match weight and inspect individual pairwise records.")

    display_cols = [
        "unique_id_l", "unique_id_r",
        "first_name_l", "first_name_r",
        "last_name_l", "last_name_r",
        "gender_l", "gender_r",
        "age_l", "age_r",
        "hba1c_l", "hba1c_r",
        "match_weight", "match_probability"
    ]
    
    st.dataframe(
        filtered_preds[display_cols].sort_values("match_weight", ascending=False),
        use_container_width=True,
        column_config={
            "match_probability": st.column_config.ProgressColumn(
                "Match Probability",
                format="%.3f",
                min_value=0.0,
                max_value=1.0
            ),
            "match_weight": st.column_config.NumberColumn("Match Weight", format="%.2f")
        }
    )

# --- TAB 2: WATERFALL DIAGNOSTICS ---
with tab2:
    st.subheader("Interactive Match Weight Waterfall Diagnostic")
    st.markdown(
        "The **Waterfall Chart** illustrates how each comparison feature contributes positive or "
        "negative evidence toward the final match score for a given pair of records."
    )

    if len(filtered_preds) > 0:
        # Create a dropdown selector for specific record pairs
        pair_options = filtered_preds.apply(
            lambda r: f"Pair [{r['unique_id_l']} ↔ {r['unique_id_r']}] | Names: ({r['first_name_l']} {r['last_name_l']} / {r['first_name_r']} {r['last_name_r']}) | Weight: {r['match_weight']:.2f}",
            axis=1
        ).tolist()

        selected_pair_str = st.selectbox("Select a Record Pair to Inspect:", pair_options)
        selected_idx = pair_options.index(selected_pair_str)
        selected_record = filtered_preds.iloc[selected_idx].to_dict()

        # Render Splink HTML Waterfall Chart via Streamlit Components iframe
        records_to_view = [selected_record]
        
        # Save temporary HTML chart using Splink's visualization engine
        chart_html_path = "temp_waterfall.html"
        linker.visualisations.waterfall_chart(records_to_view, out_path=chart_html_path)

        with open(chart_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Render embedded HTML directly in Streamlit
        components.html(html_content, height=500, scrolling=True)

    else:
        st.warning("No records pass the selected match weight threshold. Lower the threshold in the sidebar to inspect records.")

# --- TAB 3: MODEL PARAMETERS ---
with tab3:
    st.subheader("Model Parameter Diagnostics ($m$ and $u$ Probabilities)")
    st.markdown(
        "Inspect the trained Expectation-Maximization match parameters across feature levels."
    )

    chart_m_u_path = "temp_m_u.html"
    linker.visualisations.m_u_parameters_chart(out_path=chart_m_u_path)

    with open(chart_m_u_path, "r", encoding="utf-8") as f:
        m_u_html = f.read()

    components.html(m_u_html, height=700, scrolling=True)
    