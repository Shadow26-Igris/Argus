import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

DATASET_PATH = Path("./results/training_data.csv")
MODEL_PATH = Path("./models/random_forest_baseline.pkl")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# print("=" * 60)
# print("ARGUS - PHASE 2: EVALUATE BASELINE UNDER ATTACKER DRIFT")
# print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run train_baseline.py first.")

saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_cols = saved["features"]

df = pd.read_csv(DATASET_PATH)
X = df[feature_cols]
y = df["label"]
conditions = df["condition"].unique()

# 1. Feature Importance Extraction
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values(by="importance", ascending=False)

feat_imp_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
print("\nTop Predictive Features Identified by Random Forest:")
for _, row in feat_imp_df.iterrows():
    print(f"  {row['feature']:<28}: {row['importance']:.4f}")

# 2. Condition-Disaggregated Evaluation
metrics_list = []

print("\n--- Model Performance Across Attacker Adaptation Conditions ---")
for cond in ["overall", "low_change", "medium_change", "high_change"]:
    if cond == "overall":
        sub_df = df
    else:
        sub_df = df[df["condition"] == cond]
    
    sub_X = sub_df[feature_cols]
    sub_y = sub_df["label"]
    preds = model.predict(sub_X)
    
    acc = accuracy_score(sub_y, preds)
    prec = precision_score(sub_y, preds, zero_division=0)
    rec = recall_score(sub_y, preds, zero_division=0)
    f1 = f1_score(sub_y, preds, zero_division=0)
    
    cm = confusion_matrix(sub_y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics_list.append({
        "condition": cond,
        "samples": len(sub_df),
        "benign_count": (sub_y == 0).sum(),
        "attack_count": (sub_y == 1).sum(),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    })
    
    print(f"\nCondition: [{cond.upper()}] (N={len(sub_df)})")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")  #<-- (Critical for cybersecurity)"
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Attacks Detected: {tp} / {tp + fn} (Missed: {fn})")
    print(f"  False Alarms    : {fp} / {tn + fp}")

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
print(f"\n[OK] Baseline metrics saved to: {RESULTS_DIR / 'baseline_metrics.csv'}")

# 3. Generate Confusion Matrix Plot (High Change Test Set vs Overall)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Overall Matrix
overall_preds = model.predict(X)
cm_overall = confusion_matrix(y, overall_preds, labels=[0, 1])
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_overall, display_labels=["Benign (0)", "Attack (1)"])
disp1.plot(ax=axes[0], cmap=plt.cm.Blues, colorbar=False)
axes[0].set_title("Argus Baseline - Full Dataset (N=1500)")

# High Change Condition Matrix
high_mask = df["condition"] == "high_change"
high_preds = model.predict(df.loc[high_mask, feature_cols])
cm_high = confusion_matrix(df.loc[high_mask, "label"], high_preds, labels=[0, 1])
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_high, display_labels=["Benign (0)", "Attack (1)"])
disp2.plot(ax=axes[1], cmap=plt.cm.Oranges, colorbar=False)
axes[1].set_title("Argus Baseline - High Attacker Change (N=500)")

plt.tight_layout()
cm_plot_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=150)
plt.close()
print(f"[OK] Confusion matrix plot saved to: {cm_plot_path}")
