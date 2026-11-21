import pandas as pd
from pathlib import Path

# Paths
DATASET_PATH = Path("./results/training_data.csv")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# print("=" * 60)
# print("ARGUS - PHASE 2: FEATURE AUDIT & ANALYSIS")
# print("=" * 60)

df = pd.read_csv(DATASET_PATH)

# Categorization of columns based on cybersecurity observability and leakage rules
LEAKAGE_COLUMNS = [
    "selected_action",
    "executed_action",
    "behavior_type",
    "change_probability",
    "reward"
]

POSITIONAL_COLUMNS = ["episode", "step"]

CONSTANT_COLUMNS = [
    col for col in df.columns 
    if df[col].nunique() <= 1 and col not in LEAKAGE_COLUMNS + POSITIONAL_COLUMNS
]

# Telemetry features that a real defender or monitoring agent can observe
DEFENDER_FEATURES = [
    "observation_size_before",
    "observation_size_delta",
    "process_count_before",
    "process_delta",
    "session_count_before",
    "session_delta",
    "red_session_count_before",
    "red_session_delta",
    "interface_count_before",
    "interface_delta",
    "system_info_count_before",
    "system_info_delta"
]

# Ensure features actually exist in dataset
DEFENDER_FEATURES = [f for f in DEFENDER_FEATURES if f in df.columns and f not in CONSTANT_COLUMNS]

print(f"\nTotal Columns in Dataset: {len(df.columns)}")
print(f"1. Ground-Truth Leakage Columns Excluded: {len(LEAKAGE_COLUMNS)} -> {LEAKAGE_COLUMNS}")
print(f"2. Positional / Indexing Columns Excluded: {len(POSITIONAL_COLUMNS)} -> {POSITIONAL_COLUMNS}")
print(f"3. Constant / Zero-Variance Columns Excluded: {len(CONSTANT_COLUMNS)} -> {CONSTANT_COLUMNS}")
print(f"4. Selected Defender-Observable Features: {len(DEFENDER_FEATURES)} -> {DEFENDER_FEATURES}")

# Calculate correlations with label
correlations = df[DEFENDER_FEATURES + ["label"]].corr()["label"].drop("label").sort_values(ascending=False)
print("\nCorrelation with Attack Label (1 = Attack, 0 = Benign):")
for feat, corr in correlations.items():
    print(f"  {feat:<28}: {corr:+.4f}")

# Save feature documentation
feature_doc = pd.DataFrame({
    "feature": DEFENDER_FEATURES,
    "correlation_with_label": [correlations[f] for f in DEFENDER_FEATURES],
    "unique_values": [df[f].nunique() for f in DEFENDER_FEATURES],
    "mean": [df[f].mean() for f in DEFENDER_FEATURES],
    "std": [df[f].std() for f in DEFENDER_FEATURES]
})
feature_doc.to_csv(RESULTS_DIR / "feature_audit.csv", index=False)
print(f"\n[OK] Feature audit saved to: {RESULTS_DIR / 'feature_audit.csv'}")
