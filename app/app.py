import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import duckdb

from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl

# --------------------------------------------------------------------
# 1. HELPER: GENDER-ALIGNED SYNTHETIC NAME GENERATOR
# --------------------------------------------------------------------
def generate_synthetic_names(df_raw):
    male_names = [
        "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
        "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
        "Kenneth", "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
        "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon",
        "Benjamin", "Samuel", "Gregory", "Alexander", "Frank", "Patrick", "Raymond", "Jack", "Dennis", "Jerry"
    ]

    female_names = [
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
        "Lisa", "Nancy", "Betty", "Sandra", "Margaret", "Ashley", "Kimberly", "Emily", "Donna", "Michelle",
        "Carol", "Amanda", "Dorothy", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia",
        "Kathleen", "Amy", "Angela", "Shirley", "Anna", "Brenda", "Pamela", "Nicole", "Emma", "Samantha",
        "Katherine", "Christine", "Debra", "Rachel", "Carolyn", "Janet", "Maria", "Heather", "Diane", "Virginia"
    ]

    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
        "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
        "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
        "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker", "Cruz", "Edwards", "Collins", "Reyes",
        "Stewart", "Morris", "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper",
        "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
        "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
        "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross", "Foster", "Jimenez"
    ]

    df_raw["unique_id"] = df_raw["SEQN"].astype(str)

    # Match gender correctly (handles 'Male'/'Female', 1/2, or 'M'/'F')
    is_male = df_raw["gender"].astype(str).str.upper().isin(["MALE", "1", "1.0", "M"])

    np.random.seed(42)
    df_raw["first_name"] = np.where(
        is_male,
        np.random.choice(male_names, size=len(df_raw)),
        np.random.choice(female_names, size=len(df_raw))
    )
    df_raw["last_name"] = np.random.choice(last_names, size=len(df_raw))

    return df_raw


# --------------------------------------------------------------------
# 2. DATA PREPARATION (CACHED)
# --------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    csv_path = "data/nhanes_2009_2018_cardiometabolic.csv"
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
    else:
        np.random.seed(42)
        n_rows = 500
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

    # Filter to a single cycle for rapid demo execution
    if "cycle" in df_raw.columns:
        df_raw = df_raw[df_raw["cycle"] == "2009-2010"].copy()

    # Apply gender-aligned name mapping
    df_raw = generate_synthetic_names(df_raw)

    # Generate 10% test duplicates for linkage demo
    sample_size = max(5, int(len(df_raw) * 0.10))
    duplicates = df_raw.sample(n=sample_size, random_state=42).copy()
    duplicates["unique_id"] = duplicates["unique_id"] + "_dup"
    duplicates["age"] = duplicates["age"] + np.random.choice([-1, 0, 1], size=len(duplicates))

    return pd.concat([df_raw, duplicates], ignore_index=True)


# --------------------------------------------------------------------
# 3. SPLINK MODEL PIPELINE (CACHED)
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_splink_model(df_records):
    # Single-threaded DuckDB connection prevents CPU/RAM spikes
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=1;")
    db_api = DuckDBAPI(connection=con)

    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            block_on("first_name", "gender"),
            block_on("last_name", "gender"),
            block_on("first_name", "last_name"),
        ],
        comparisons=[
            cl.ExactMatch("first_name"),
            cl.ExactMatch("last_name"),
            cl.ExactMatch("gender"),
            cl.ExactMatch("age"),
        ],
        probability_two_random_records_match=0.01,
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True
    )

    linker = Linker(df_records, settings, db_api=db_api)
    
    # Fast parameter estimation via random sampling
    linker.training.estimate_u_using_random_sampling(max_pairs=1000, seed=42)

    # Execute predictions
    predictions = linker.inference.predict()
    
    df_preds = predictions.as_pandas_dataframe()
    records_dict = predictions.as_record_dict()

    m_u_chart = linker.visualisations.m_u_parameters_chart()
    m_u_html = m_u_chart.as_html() if hasattr(m_u_chart, "as_html") else str(m_u_chart)

    con.close()
    return df_preds, records_dict, m_u_html


# --------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiometabolic Entity Linkage",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Cardiometabolic Entity Linkage Dashboard")
st.markdown("Probabilistic record linkage on NHANES clinical survey data using **Splink** & **DuckDB**.")

df_records = load_and_prepare_data()

with st.spinner("⚡ Running Splink Linkage Pipeline..."):
    df_predictions, records_dict, m_u_html = run_splink_model(df_records)

# Metrics Summary Bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Input Records", f"{len(df_records):,}")
col2.metric("Candidate Pairs Evaluated", f"{len(df_predictions):,}")
high_prob_matches = len(df_predictions[df_predictions["match_probability"] >= 0.8])
col3.metric("High-Confidence Matches (≥80%)", f"{high_prob_matches:,}")
avg_prob = df_predictions["match_probability"].mean() if not df_predictions.empty else 0
col4.metric("Average Match Probability", f"{avg_prob:.1%}")

st.markdown("---")

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["📋 Linked Predictions", "📊 Model Parameters", "🔍 Match Inspection"])

with tab1:
    st.subheader("Predicted Pair Matches")
    min_prob = st.slider("Filter by Minimum Match Probability:", 0.0, 1.0, 0.5, 0.05)
    
    df_filtered = df_predictions[df_predictions["match_probability"] >= min_prob].copy()
    
    display_cols = [col for col in [
        "match_weight", "match_probability", 
        "first_name_l", "first_name_r", 
        "last_name_l", "last_name_r", 
        "age_l", "age_r", 
        "gender_l", "gender_r", 
        "unique_id_l", "unique_id_r"
    ] if col in df_filtered.columns]
    
    st.dataframe(df_filtered[display_cols], use_container_width=True)

with tab2:
    st.subheader("m and u Parameter Weights")
    st.markdown("Estimated match ($m$) and unmatch ($u$) probabilities across fields.")
    components.html(m_u_html, height=500, scrolling=True)

with tab3:
    st.subheader("Match Probability Breakdown")
    if records_dict:
        selected_index = st.selectbox(
            "Select Record Pair Index to Inspect:",
            options=range(min(len(records_dict), 50)),
            format_func=lambda i: f"Pair {i}: {records_dict[i].get('first_name_l', '')} {records_dict[i].get('last_name_l', '')} <-> {records_dict[i].get('first_name_r', '')} {records_dict[i].get('last_name_r', '')} (Prob: {records_dict[i].get('match_probability', 0):.2f})"
        )
        
        st.json(records_dict[selected_index])
    else:
        st.info("No record pairs available for inspection.")
