import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATASET_PATH = Path("./results/training_data.csv")
MODELS_DIR = Path("./models")
RESULTS_DIR = Path("./results")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# print("=" * 60)
# print("ARGUS - PHASE 2: TRAIN STATIC RANDOM FOREST BASELINE")
# print("=" * 60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

# Exclude constant features dynamically
constant_cols = [c for c in df.columns if df[c].nunique() <= 1]

# Defender-observable telemetry candidate features
CANDIDATE_FEATURES = [
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

FEATURE_COLUMNS = [f for f in CANDIDATE_FEATURES if f in df.columns and f not in constant_cols]
TARGET_COLUMN = "label"

print(f"\nFeatures selected for training ({len(FEATURE_COLUMNS)}):")
for feat in FEATURE_COLUMNS:
    print(f"  - {feat}")

# Condition-based Train/Test Split
# Train on Low + Medium Change (1,000 samples)
# Test on High Change (500 samples) to evaluate resilience against attacker adaptation
train_mask = df["condition"].isin(["low_change", "medium_change"])
test_mask = df["condition"] == "high_change"

X_train = df.loc[train_mask, FEATURE_COLUMNS]
y_train = df.loc[train_mask, TARGET_COLUMN]

X_test = df.loc[test_mask, FEATURE_COLUMNS]
y_test = df.loc[test_mask, TARGET_COLUMN]

print(f"\nDataset Splitting:")
print(f"  Training samples (Low + Medium Change): {len(X_train)} (Benign: {(y_train == 0).sum()}, Attack: {(y_train == 1).sum()})")
print(f"  Testing samples  (High Change)         : {len(X_test)} (Benign: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()})")

# Initialize and train Random Forest Classifier
print("\nTraining Random Forest baseline (n_estimators=100, max_depth=8, random_state=42)...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Save model artifact
MODEL_PATH = MODELS_DIR / "random_forest_baseline.pkl"
joblib.dump({
    "model": model,
    "features": FEATURE_COLUMNS
}, MODEL_PATH)

print(f"[OK] Trained model artifact saved to: {MODEL_PATH}")

# In-sample training performance
y_train_pred = model.predict(X_train)
print("\n--- In-Sample Training Performance (Low + Med Change) ---")
print(f"  Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")
print(f"  Precision: {precision_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"  F1-Score : {f1_score(y_train, y_train_pred, zero_division=0):.4f}")

# Quick out-of-distribution preview
y_test_pred = model.predict(X_test)
print("\n--- Out-of-Distribution Test Performance (High Change) ---")
print(f"  Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_test_pred, zero_division=0):.4f}")
