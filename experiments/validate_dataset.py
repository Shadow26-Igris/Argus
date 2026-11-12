import pandas as pd
from pathlib import Path

DATASET_PATH = Path("./results/training_data.csv")

EXPECTED_RECORDS = 1500

REQUIRED_COLUMNS = [
    "episode",
    "step",
    "visible_hosts_before",
    "visible_hosts_after",
    "interface_count_before",
    "interface_count_after",
    "system_info_count_before",
    "system_info_count_after",
    "observation_keys_before",
    "observation_keys_after",
    "observation_size_before",
    "observation_size_after",
    "success_before",
    "success_after",
    "label",
    "executed_action",
    "condition",
    "change_probability",
    "process_count_before",
    "process_count_after",
    "process_delta",
    "session_count_before",
    "session_count_after",
    "session_delta",
    "red_session_count_before",
    "red_session_count_after",
    "red_session_delta",
]


def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
        return True
    else:
        print(f"[FAIL] {message}")
        return False


print("=" * 60)
print("ARGUS - PHASE 1 DATASET VALIDATION")
print("=" * 60)

if not DATASET_PATH.exists():
    print(f"\n[ERROR] Dataset not found: {DATASET_PATH}")
    raise SystemExit(1)

df = pd.read_csv(DATASET_PATH)

print(f"\nDataset: {DATASET_PATH}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

checks = []

# --------------------------------------------------
# 1. Record count
# --------------------------------------------------

checks.append(
    check(
        len(df) == EXPECTED_RECORDS,
        f"Expected {EXPECTED_RECORDS} records, found {len(df)}"
    )
)

# --------------------------------------------------
# 2. Required columns
# --------------------------------------------------

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

checks.append(
    check(
        len(missing_columns) == 0,
        "All required telemetry columns are present"
        if not missing_columns
        else f"Missing columns: {missing_columns}"
    )
)

# --------------------------------------------------
# 3. Label validation
# --------------------------------------------------

unique_labels = sorted(df["label"].dropna().unique().tolist())

checks.append(
    check(
        set(unique_labels).issubset({0, 1}),
        f"Labels contain only 0/1: {unique_labels}"
    )
)

# --------------------------------------------------
# 4. Missing values
# --------------------------------------------------

# previous_action is intentionally allowed to be missing
# for the first step of each condition block because there
# is no previous action yet.

analysis_columns_allowed_missing = [
    "previous_action"
]

missing_by_column = df.isna().sum()

unexpected_missing = {
    column: int(count)
    for column, count in missing_by_column.items()
    if count > 0 and column not in analysis_columns_allowed_missing
}

allowed_missing = {
    column: int(count)
    for column, count in missing_by_column.items()
    if count > 0 and column in analysis_columns_allowed_missing
}

print("\nMissing values:")
if allowed_missing:
    for column, count in allowed_missing.items():
        print(f"  Allowed: {column} = {count}")

if unexpected_missing:
    for column, count in unexpected_missing.items():
        print(f"  Unexpected: {column} = {count}")

checks.append(
    check(
        len(unexpected_missing) == 0,
        "No unexpected missing values found"
    )
)

# --------------------------------------------------
# 5. Attack / benign distribution
# --------------------------------------------------

label_counts = df["label"].value_counts().sort_index()

benign_count = int(label_counts.get(0, 0))
attack_count = int(label_counts.get(1, 0))

print("\nLabel distribution:")
print(f"  Benign (0): {benign_count}")
print(f"  Attack (1): {attack_count}")

checks.append(
    check(
        benign_count > 0 and attack_count > 0,
        "Both benign and attack samples are present"
    )
)

# --------------------------------------------------
# 6. Dataset conditions
# --------------------------------------------------

expected_conditions = {
    "low_change",
    "medium_change",
    "high_change",
}

actual_conditions = set(df["condition"].unique())

print("\nAttacker-change conditions:")
for condition in sorted(actual_conditions):
    count = int((df["condition"] == condition).sum())
    print(f"  {condition}: {count}")

checks.append(
    check(
        expected_conditions.issubset(actual_conditions),
        "Low, medium and high attacker-change conditions are present"
    )
)

# --------------------------------------------------
# 7. Attack action diversity
# --------------------------------------------------

action_counts = df["executed_action"].value_counts()

print("\nExecuted actions:")
for action, count in action_counts.items():
    print(f"  {action}: {count}")

attack_actions = [
    "ExploitDroneVulnerability",
    "SeizeControl",
    "FloodBandwidth",
    "BlockTraffic",
]

present_attack_actions = [
    action for action in attack_actions
    if action in action_counts.index
]

checks.append(
    check(
        len(present_attack_actions) >= 3,
        f"Attack action diversity is sufficient: "
        f"{len(present_attack_actions)} attack action types found"
    )
)

# --------------------------------------------------
# 8. Unknown labels / invalid labels
# --------------------------------------------------

unknown_labels = df[
    ~df["label"].isin([0, 1])
]

checks.append(
    check(
        len(unknown_labels) == 0,
        "No unknown/invalid training labels are present"
    )
)

# --------------------------------------------------
# 9. Telemetry variation
# --------------------------------------------------

telemetry_columns = [
    "visible_hosts_before",
    "observation_size_before",
    "process_count_before",
    "session_count_before",
    "red_session_count_before",
]

print("\nTelemetry variation:")

constant_features = []
variable_features = []

for col in telemetry_columns:
    if col in df.columns:
        unique_count = df[col].nunique()

        print(f"  {col}: {unique_count} unique values")

        if unique_count <= 1:
            constant_features.append(col)
        else:
            variable_features.append(col)

print("\nConstant telemetry features:")
if constant_features:
    for col in constant_features:
        print(f"  {col}")

print("\nVariable telemetry features:")
if variable_features:
    for col in variable_features:
        print(f"  {col}")

checks.append(
    check(
        len(variable_features) > 0,
        "At least one telemetry feature contains meaningful variation"
    )
)
# --------------------------------------------------
# 10. Process/session telemetry
# --------------------------------------------------

checks.append(
    check(
        "process_count_before" in df.columns
        and "process_count_after" in df.columns,
        "Process telemetry is available"
    )
)

checks.append(
    check(
        "session_count_before" in df.columns
        and "session_count_after" in df.columns,
        "Session telemetry is available"
    )
)

checks.append(
    check(
        "red_session_count_before" in df.columns
        and "red_session_count_after" in df.columns,
        "Red-session telemetry is available"
    )
)

# --------------------------------------------------
# 11. Episode structure
# --------------------------------------------------

if "episode" in df.columns:

    episode_count = df["episode"].nunique()
    steps_per_episode = df.groupby("episode")["step"].count()

    print("\nEpisode structure:")
    print(f"  Episodes: {episode_count}")
    print(f"  Steps per episode:")
    print(steps_per_episode.to_string())

    checks.append(
        check(
            episode_count == 5,
            f"Expected 15 episodes, found {episode_count}"
        )
    )

# --------------------------------------------------
# 12. Data leakage check
# --------------------------------------------------

leakage_columns = [
    "selected_action",
    "executed_action",
    "behavior_type",
    "change_probability",
    "reward",
]

print("\nPotential ML leakage columns:")
for col in leakage_columns:
    if col in df.columns:
        print(f"  {col}")

checks.append(
    check(
        all(col in df.columns for col in leakage_columns),
        "Attack metadata columns are available for analysis and can be excluded from ML"
    )
)

# --------------------------------------------------
# Final result
# --------------------------------------------------

passed = sum(checks)
total = len(checks)

print("\n" + "=" * 60)
print(f"VALIDATION RESULT: {passed}/{total} CHECKS PASSED")
print("=" * 60)

if passed == total:
    print("\nPHASE 1 DATASET VALIDATION: PASSED")
    print("The dataset is ready for Phase 2 ML baseline development.")
else:
    print("\nPHASE 1 DATASET VALIDATION: FAILED")
    print("Fix the failed checks before starting ML training.")