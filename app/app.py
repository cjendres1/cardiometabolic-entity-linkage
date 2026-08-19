import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import random
import os
import duckdb

from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl

# --------------------------------------------------------------------
# HELPER: SYNTHETIC NAME GENERATOR
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

    # Align gender
    is_male = df_raw["gender"].astype(str).str.upper().isin(["1", "M", "MALE", "1.0"])
    
    np.random.seed(42)  # Fixed seed for consistent rendering across reloads
    df_raw["first_name"] = np.where(
        is_male,
        np.random.choice(male_names, size=len(df_raw)),
        np.random.choice(female_names, size=len(df_raw))
    )
    df_raw["last_name"] = np.random.choice(last_names, size=len(df_raw))

    return df_raw

# -----------------------------------------------------------------------------
# CONFIG & PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiometabolic Entity Linker",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Cardiometabolic Entity Linkage Dashboard")

# --------------------------------------------------------------------
# DATA LOADING FUNCTION
# --------------------------------------------------------------------
@st.cache_data
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

    # Apply gender-aligned name mapping with expanded distributions
    df_raw = generate_synthetic_names(df_raw)

    # Generate test duplicate records for linkage demo
    sample_size = max(5, int(len(df_raw) * 0.10))
    duplicates = df_raw.sample(n=sample_size, random_state=42).copy()
    duplicates["unique_id"] = duplicates["unique_id"] + "_dup"
    duplicates["age"] = duplicates["age"] + np.random.choice([-1, 0, 1], size=len(duplicates))

    return pd.concat([df_raw, duplicates], ignore_index=True)

@st.cache_data(show_spinner=False)
def run_splink_model(df_records):
    # Single-threaded DuckDB to prevent container healthcheck timeouts
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
            cl.LevenshteinAtThresholds("first_name", [1, 2]),
            cl.LevenshteinAtThresholds("last_name", [1, 2]),
            cl.ExactMatch("gender"),
            cl.ExactMatch("age"),
        ],
        probability_two_random_records_match=0.01
    )

    linker = Linker(df_records, settings, db_api=db_api)
    
    # Fast u-probability estimation via random sampling (< 0.2s)
    linker.training.estimate_u_using_random_sampling(max_pairs=500, seed=42)

    # Skip EM parameter training loops entirely on synthetic demo data
    # Splink will automatically apply standard default weights for m parameters

    # Fast prediction (< 1s execution)
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
