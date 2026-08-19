import pandas as pd
import numpy as np
import random
from splink.duckdb.linker import DuckDBLinker
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl
from splink.duckdb.blocking_rule_library import block_on

# 1. Load Cleaned NHANES Data
df_raw = pd.read_csv("data/nhanes_2009_2018_cardiometabolic.csv")

# Generate synthetic identifiers (First/Last names) for linkage evaluation
first_names = ["James", "John", "Robert", "Michael", "William", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

np.random.seed(42)
df_raw["unique_id"] = df_raw["SEQN"].astype(str)
df_raw["first_name"] = np.random.choice(first_names, size=len(df_raw))
df_raw["last_name"] = np.random.choice(last_names, size=len(df_raw))

# 2. Inject Synthetic Noise (Create duplicate pairs with typos)
sample_size = int(len(df_raw) * 0.10)
duplicates = df_raw.sample(n=sample_size, random_state=42).copy()

# Add typos / phonetic modifications to duplicates
def inject_typo(name):
    if len(name) > 3 and random.random() > 0.5:
        idx = random.randint(0, len(name) - 1)
        return name[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + name[idx+1:]
    return name

duplicates["unique_id"] = duplicates["unique_id"] + "_dup"
duplicates["first_name"] = duplicates["first_name"].apply(inject_typo)
duplicates["age"] = duplicates["age"] + np.random.choice([-1, 0, 1], size=len(duplicates))
duplicates["hba1c"] = np.round(duplicates["hba1c"] + np.random.normal(0, 0.2, size=len(duplicates)), 1)

# Combined dataframe for deduplication / linkage
df_concat = pd.concat([df_raw, duplicates], ignore_index=True)

# 3. Configure Splink Model Settings
settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        block_on("first_name", "gender"),
        block_on("last_name", "gender"),
        block_on("age", "gender")
    ],
    "comparisons": [
        # Name comparisons using Jaro-Winkler distance
        ctl.name_comparison("first_name", jaro_winkler_thresholds=[0.88, 0.94]),
        ctl.name_comparison("last_name", jaro_winkler_thresholds=[0.88, 0.94]),
        
        # Exact matching on categorical demographic features
        cl.exact_match("gender"),
        
        # Numeric range comparison for age
        cl.numeric_difference_at_thresholds("age", thresholds=[1, 3, 5]),
        
        # Clinical lab parameter threshold comparison (HbA1c)
        cl.numeric_difference_at_thresholds("hba1c", thresholds=[0.2, 0.5, 1.0]),
    ],
    "retain_matching_framework": True,
    "retain_intermediate_calculation_columns": True
}

# 4. Initialize DuckDB Linker & Train Model via Expectation-Maximization
linker = DuckDBLinker(df_concat, settings)

# Estimate m and u probabilities using unsupervised EM
linker.estimate_u_probability_two_random_records_match(
    max_pairs=100_000, 
    seed=42
)

linker.estimate_parameters_using_expectation_maximization(
    block_on("first_name")
)
linker.estimate_parameters_using_expectation_maximization(
    block_on("last_name")
)

# 5. Predict Matches & Extract High-Probability Links
predictions = linker.predict(match_weight_threshold=2.0)
df_predictions = predictions.as_pandas_dataframe()

print(f"Probabilistic Linkage Complete.")
print(f"Total Candidate Pairs Evaluated: {len(df_predictions)}")
print(df_predictions[["unique_id_l", "unique_id_r", "match_weight", "match_probability"]].head(10))

# Save waterfall diagnostics HTML chart
linker.visualisations.waterfall_chart(
    df_predictions.to_dict(orient="records")[:5], 
    out_path="data/splink_waterfall_diagnostic.html"
)
