import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import random
import os
import duckdb

# --- Splink Version-Agnostic Imports ---
try:
    from splink import Linker, DuckDBAPI, block_on
    import splink.comparison_library as cl
except (ImportError, ModuleNotFoundError):
    try:
        from splink import Linker, DuckDBAPI, block_on
        import splink.comparisons as cl
    except (ImportError, ModuleNotFoundError):
        from splink.duckdb.linker import DuckDBLinker as Linker
        from splink.duckdb.blocking_rule_library import block_on
        import splink.duckdb.comparison_library as cl

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
    "the **Fellegi-Sunter model** (`Splink 4` + `DuckDB`)."
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
        # Fallback mock data if CSV is not present
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

    # Synthetic identities for linkage demonstration
    first_names = ["James", "John", "Robert", "Michael", "William", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

    np.random.seed(42)
    df_raw["unique_id"] = df_raw["SEQN"].astype(str)
    df_raw["first_name"] = np.random.choice(first_names, size=len(df_raw))
    df_raw["last_name"] = np.random.choice(last_names, size=len(df_raw))

    # Inject 10% duplicates with noise
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

@st.cache_data
def run_splink_pipeline(df_records):
    """Executes Splink linkage safely, extracts prediction data and html charts, and closes DuckDB."""
    con = duckdb.connect()
    con.execute("SET threads = 1;")
    db_api = DuckDBAPI(connection=con)

    settings = {
        "link_type": "dedupe_only",
        "blocking_rules_to_generate_predictions": [
            block_on("first_name", "gender"),
            block_on("last_name", "gender"),
            block_on("age", "gender")
        ],
        "comparisons": [
            cl.LevenshteinAtThresholds("first_name", 2),
            cl.LevenshteinAtThresholds("last_name", 2),
            cl.ExactMatch("gender"),
            cl.NumericDifferenceAtThresholds("age", thresholds=[1, 3, 5]),
            cl.NumericDifferenceAtThresholds("hba1c", thresholds=[0.2, 0.5, 1.0]),
        ],
        "retain_matching_framework": True,
        "retain_intermediate_calculation_columns": True
    }

    linker = Linker(df_records, settings, db_api=db_api)
    
    # Fast parameter estimation
    linker.training.estimate_u_probability_two_random_records_match(max_pairs=2_500, seed=42)
    linker.training.estimate_parameters_using_expectation_maximization(
        block_on("first_name"), max_iterations=3
    )
    linker.training.estimate_parameters_using_expectation_maximization(
        block_on("last_name"), max_iterations=3
    )

    predictions = linker.inference.predict(match_weight_threshold=-5.0)
    df_preds = predictions.as_pandas_dataframe()
    records_dict = predictions.as_record_dict()

    # Pre-render m/u parameters chart HTML
    chart_m_u_path = "temp_m_u.html"
    linker.visualisations.m_u_parameters_chart(out_path=chart_m_u_path)
    with open(chart_m_u_path, "r", encoding="utf-8") as f:
        m_u_html = f.read()

    # Pre-render waterfall chart HTML for top predicted match pairs
    waterfall_html_map = {}
    for idx, record in enumerate(records_dict[:50]):  # Cache top 50 pairs for fast UI rendering
        pair_key = f"{record.get('unique_id_l')}---{record.get('unique_id_r')}"
        chart_path = f"temp_waterfall_{idx}.html"
        linker.visualisations.waterfall_chart([record], out_path=chart_path)
        with open(chart_path, "r", encoding="utf-8") as f:
            waterfall_html_map[pair_key] = f.read()

    con.close()
    return df_preds, records_dict, m_u_html, waterfall_html_map

# Load Data and Run Pipeline
df_records = load_and_prepare_data()

with st.spinner("⚡ Running Splink linkage pipeline (3–5 seconds)..."):
    df_predictions, records_dict, m_u_html, waterfall_html_map = run_splink_pipeline(df_records)

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
        pair_options = filtered_preds.apply(
            lambda r: (
                f"Pair [{r['unique_id_l']} ↔ {r['unique_id_r']}] | "
                f"Names: ({r['first_name_l']} {r['last_name_l']} / {r['first_name_r']} {r['last_name_r']}) | "
                f"Weight: {r['match_weight']:.2f}"
            ),
            axis=1
        ).tolist()

        selected_pair_str = st.selectbox("Select a Record Pair to Inspect:", pair_options)
        selected_row = filtered_preds.iloc[pair_options.index(selected_pair_str)]
        lookup_key = f"{selected_row['unique_id_l']}---{selected_row['unique_id_r']}"

        if lookup_key in waterfall_html_map:
            components.html(waterfall_html_map[lookup_key], height=500, scrolling=True)
        else:
            st.info("Waterfall diagnostic chart is available for the top candidate pairs.")
    else:
        st.warning("No records pass the selected match weight threshold. Lower the threshold in the sidebar.")

# --- TAB 3: MODEL PARAMETERS ---
with tab3:
    st.subheader("Model Parameter Diagnostics ($m$ and $u$ Probabilities)")
    st.markdown("Inspect the trained Expectation-Maximization match parameters across feature levels.")

    components.html(m_u_html, height=700, scrolling=True)
    