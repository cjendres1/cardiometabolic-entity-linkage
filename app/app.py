import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import random
import os
import duckdb

from splink import Linker, DuckDBAPI, SettingsCreator, block_on
import splink.comparison_library as cl

# -----------------------------------------------------------------------------
# CONFIG & PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiometabolic Entity Linker",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Cardiometabolic Entity Linkage Dashboard")

# -----------------------------------------------------------------------------
# CACHED DATA & LINKAGE PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    csv_path = "data/nhanes_2009_2018_cardiometabolic.csv"
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
    else:
        np.random.seed(42)
        n_rows = 500  # Scaled down for fast demo rendering
        df_raw = pd.DataFrame({
            "SEQN": range(100000, 100000 + n_rows),
            "age": np.random.randint(18, 80, size=n_rows),
            "gender": np.random.choice(["Male", "Female"], size=n_rows),
            "hba1c": np.round(np.random.normal(5.7, 1.2, size=n_rows), 1),
            "fasting_glucose": np.random.randint(70, 180, size=n_rows),
            "sys_bp": np.random.randint(110, 160, size=n_rows),
            "diabetes_status": np.random.choice(["Yes", "No", "Borderline"], size=n_rows),
            "cycle": np.random.choice(["2009-2010", "2011-2012", "2013-2014"], size=n_rows)
        })

    first_names = ["James", "John", "Robert", "Michael", "William", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

    df_raw["unique_id"] = df_raw["SEQN"].astype(str)
    df_raw["first_name"] = np.random.choice(first_names, size=len(df_raw))
    df_raw["last_name"] = np.random.choice(last_names, size=len(df_raw))

    sample_size = max(5, int(len(df_raw) * 0.10))
    duplicates = df_raw.sample(n=sample_size, random_state=42).copy()
    duplicates["unique_id"] = duplicates["unique_id"] + "_dup"
    duplicates["age"] = duplicates["age"] + np.random.choice([-1, 0, 1], size=len(duplicates))

    return pd.concat([df_raw, duplicates], ignore_index=True)

@st.cache_data(show_spinner=False)
def run_splink_model(df_records):
    # Isolated in-memory DuckDB connection
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads = 1;")
    db_api = DuckDBAPI(connection=con)

    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            block_on("first_name", "gender"),
            block_on("last_name", "gender")
        ],
        comparisons=[
            cl.LevenshteinAtThresholds("first_name", [1, 2]),
            cl.LevenshteinAtThresholds("last_name", [1, 2]),
            cl.ExactMatch("gender"),
            cl.ExactMatch("age"),
        ],
        probability_two_random_records_match=0.01
    )

    linker = Linker(df_records, settings, db_api=db_api)
    
    # 1. Estimate u probabilities via random sampling
    linker.training.estimate_u_using_random_sampling(max_pairs=1000, seed=42)
    
    # 2. Train m probabilities across complementary blocking rules
    # (first_name trains last_name/gender; last_name trains first_name/gender)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("first_name"),
        estimate_without_term_frequencies=True
    )
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("last_name"),
        estimate_without_term_frequencies=True
    )

    # 3. Predict without blocking on full cartesian product unnecessarily
    predictions = linker.inference.predict()
    df_preds = predictions.as_pandas_dataframe()
    records_dict = predictions.as_record_dict()

    m_u_chart = linker.visualisations.m_u_parameters_chart()
    m_u_html = m_u_chart.as_html() if hasattr(m_u_chart, "as_html") else str(m_u_chart)

    con.close()
    return df_preds, records_dict, m_u_html

# Execute Pipeline safely
df_records = load_and_prepare_data()

with st.spinner("⚡ Running Splink Linkage Pipeline..."):
    df_predictions, records_dict, m_u_html = run_splink_model(df_records)

# -----------------------------------------------------------------------------
# INTERFACE
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Linkage Filters")
min_weight = float(df_predictions["match_weight"].min())
max_weight = float(df_predictions["match_weight"].max())

weight_threshold = st.sidebar.slider(
    "Match Weight Threshold",
    min_value=round(min_weight, 1),
    max_value=round(max_weight, 1),
    value=1.0,
    step=0.5
)

filtered_preds = df_predictions[df_predictions["match_weight"] >= weight_threshold].copy()

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(df_records))
c2.metric("Evaluated Pairs", len(df_predictions))
c3.metric("Predicted Links", len(filtered_preds))

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Predicted Links", "🌊 On-Demand Waterfall Diagnostic", "📈 Model Parameters"])

with tab1:
    st.dataframe(filtered_preds, use_container_width=True)

with tab2:
    st.subheader("On-Demand Waterfall Diagnostic")
    if len(filtered_preds) > 0:
        pair_map = {
            f"Pair [{r['unique_id_l']} ↔ {r['unique_id_r']}] | Weight: {r['match_weight']:.2f}": idx 
            for idx, r in filtered_preds.head(20).iterrows()
        }
        selected_label = st.selectbox("Select Record Pair:", list(pair_map.keys()))
        selected_idx = pair_map[selected_label]
        
        # Render specific waterfall on demand rather than pre-computing 30 in a loop
        record_match = [r for r in records_dict if r.get('unique_id_l') == filtered_preds.loc[selected_idx, 'unique_id_l'] and r.get('unique_id_r') == filtered_preds.loc[selected_idx, 'unique_id_r']]
        
        if record_match:
            con_temp = duckdb.connect(database=":memory:")
            db_api_temp = DuckDBAPI(connection=con_temp)
            linker_temp = Linker(df_records, SettingsCreator(link_type="dedupe_only", comparisons=[cl.ExactMatch("gender")]), db_api=db_api_temp)
            
            wf_chart = linker_temp.visualisations.waterfall_chart(record_match)
            html_val = wf_chart.as_html() if hasattr(wf_chart, "as_html") else str(wf_chart)
            components.html(html_val, height=500, scrolling=True)
            con_temp.close()
    else:
        st.warning("No records meet threshold criteria.")

with tab3:
    components.html(m_u_html, height=650, scrolling=True)
