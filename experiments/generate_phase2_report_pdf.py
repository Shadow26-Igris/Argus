import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_PDF = Path("./Argus_Phase_2_Report.pdf")
RESULTS_DIR = Path("./results")
CM_IMAGE = RESULTS_DIR / "confusion_matrix.png"

with PdfPages(OUTPUT_PDF) as pdf:
    # =========================================================================
    # PAGE 1: TITLE, METADATA & EXECUTIVE SUMMARY
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Title & Header
    ax.text(0.5, 0.95, "PROJECT ARGUS", fontsize=24, fontweight='bold', ha='center', color='#0d3b66')
    ax.text(0.5, 0.915, "Phase 2 Technical Report: Static ML Baseline & Attacker Drift Analysis", 
            fontsize=12, fontweight='bold', ha='center', color='#2b2d42')
    ax.axhline(y=0.895, xmin=0.08, xmax=0.92, color='#0d3b66', linewidth=2.5)
    
    # Metadata Box
    meta_box = dict(boxstyle='round,pad=0.6', facecolor='#edf2f4', edgecolor='#8d99ae', linewidth=1)
    meta_str = (
        "Environment : CybORG Drone Swarm Simulation           |   Model   : Balanced Random Forest Baseline\n"
        "Dataset     : 1,500 Telemetry Records (5 Episodes)    |   Target  : Binary Classification (0=Benign, 1=Attack)\n"
        "Phase Scope : Static Attack Detection Baseline        |   Git Ref : Commit ed5b9a9 (Phase 1 Baseline)"
    )
    ax.text(0.5, 0.835, meta_str, fontsize=8.5, ha='center', va='center', color='#2b2d42', 
            family='monospace', bbox=meta_box)
    
    # Section 1: Executive Summary
    ax.text(0.08, 0.76, "1. Executive Summary & Objective", fontsize=13, fontweight='bold', color='#0d3b66')
    exec_summary = (
        "Phase 2 marks the critical transition of Project Argus from environment configuration and data validation into\n"
        "machine learning modeling. The primary objective was to build a rigorous, static attack detection baseline using\n"
        "defender-observable telemetry from the CybORG drone-swarm scenario, and empirically evaluate whether\n"
        "conventional machine learning remains effective when the red attacker adapts its behavior over time.\n\n"
        "Key Deliverable Outcome:\n"
        "A Random Forest classifier (100 estimators, max depth 8, balanced class weighting) was trained exclusively\n"
        "on Low-Change and Medium-Change conditions (1,000 samples) and evaluated out-of-distribution on unseen\n"
        "High-Change conditions (500 samples). The evaluation revealed a sharp performance collapse: accuracy dropped\n"
        "from 94.40% under low adaptation down to 67.20% under high adaptation, driven by a catastrophic surge in\n"
        "false alarms (84.1% false-positive rate)."
    )
    ax.text(0.08, 0.60, exec_summary, fontsize=9, color='#2b2d42', linespacing=1.35)
    
    # Section 2: Core Research Question Answered
    ax.text(0.08, 0.54, "2. Core Research Question Answered", fontsize=13, fontweight='bold', color='#0d3b66')
    rq_text = (
        "The overarching research question of Project Argus is:\n"
        "   'Can an ML-based defense remain effective when the attacker changes its behavior over time?'\n\n"
        "Empirical Answer from Phase 2:\n"
        "   NO. While standard static ML detects static attacks effectively (94.40% accuracy), it cannot generalize\n"
        "   to adaptive adversaries. Attacker behavioral shifts induce severe concept drift, proving that static ML\n"
        "   defenses are fragile and justifying the need for continuous adaptive / online learning pipelines in Phase 3."
    )
    ax.text(0.08, 0.40, rq_text, fontsize=9, color='#2b2d42', linespacing=1.35)
    
    # Section 3: Summary of Project Modifications & Files Created
    ax.text(0.08, 0.34, "3. Summary of Project Modifications & Files Created", fontsize=13, fontweight='bold', color='#0d3b66')
    files_table = [
        ["File / Artifact", "Type", "Description / Role in Phase 2"],
        ["experiments/feature_analysis.py", "Script", "Feature hygiene audit; quarantined leakage and zero-variance columns."],
        ["experiments/train_baseline.py", "Script", "Condition-aware train/test split; trained balanced Random Forest baseline."],
        ["experiments/evaluate_baseline.py", "Script", "Evaluated model per condition; exported metrics and confusion plots."],
        ["models/random_forest_baseline.pkl", "Model", "Serialized model artifact with trained weights and feature schema."],
        ["results/feature_audit.csv", "Data", "Complete audit of all 36 columns with exclusion rationales."],
        ["results/feature_importance.csv", "Data", "Relative feature importance rankings derived from the forest."],
        ["results/baseline_metrics.csv", "Data", "Disaggregated Accuracy, Precision, Recall, F1, and Confusion counts."],
        ["results/confusion_matrix.png", "Plot", "High-resolution side-by-side comparative confusion matrix plot."]
    ]
    t_files = ax.table(cellText=files_table[1:], colLabels=files_table[0], loc='center',
                       cellLoc='left', bbox=[0.08, 0.10, 0.84, 0.22])
    t_files.auto_set_font_size(False)
    t_files.set_fontsize(7.5)
    for (row, col), cell in t_files.get_celld().items():
        if row == 0:
            cell.set_facecolor('#0d3b66')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else 'white')
            
    ax.text(0.5, 0.04, "Page 1 of 4   |   Project Argus — Phase 2 Technical Report", fontsize=8, ha='center', color='#8d99ae')
    pdf.savefig(fig, dpi=150)
    plt.close()

    # =========================================================================
    # PAGE 2: FEATURE HYGIENE & SELECTION AUDIT
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    ax.text(0.5, 0.95, "FEATURE HYGIENE & SELECTION AUDIT", fontsize=18, fontweight='bold', ha='center', color='#0d3b66')
    ax.axhline(y=0.92, xmin=0.08, xmax=0.92, color='#0d3b66', linewidth=2)
    
    ax.text(0.08, 0.88, "1. Ground-Truth Leakage Isolation (Scientific Rigor)", fontsize=13, fontweight='bold', color='#0d3b66')
    leakage_desc = (
        "In simulated cyber environments like CybORG, internal engine states track attacker choices directly.\n"
        "Feeding these into an ML model causes catastrophic data leakage (falsely inflating accuracy to ~100%).\n"
        "In Phase 2, we conducted a comprehensive feature audit and strictly quarantined all leakage columns:"
    )
    ax.text(0.08, 0.81, leakage_desc, fontsize=9, color='#2b2d42', linespacing=1.3)
    
    leakage_table = [
        ["Excluded Column", "Category", "Reason for Exclusion from ML Features"],
        ["selected_action", "Attacker Intent", "Directly names the specific exploit selected by the red agent"],
        ["executed_action", "Attacker Execution", "Ground-truth execution label (e.g. FloodBandwidth, SeizeControl)"],
        ["behavior_type", "Attacker Policy", "Categorical identifier of the red agent's behavioral strategy"],
        ["change_probability", "Hyperparameter", "Internal engine probability controlling attacker adaptation rate"],
        ["reward", "Game State", "Simulator scoring metric, completely unavailable to real-world defenders"],
        ["episode, step", "Index Metadata", "Positional temporal coordinates, non-telemetric"],
        ["visible_hosts_before/after", "Zero Variance", "Constant at 18 across all 1,500 rows; zero predictive value"],
        ["interface_count (before/after/delta)", "Zero Variance", "Constant at 18 across all rows; provides no discriminating signal"],
        ["system_info_count (before/after/delta)", "Zero Variance", "Constant at 18 across all rows; provides no discriminating signal"]
    ]
    t_leakage = ax.table(cellText=leakage_table[1:], colLabels=leakage_table[0], loc='center',
                         cellLoc='left', bbox=[0.08, 0.53, 0.84, 0.26])
    t_leakage.auto_set_font_size(False)
    t_leakage.set_fontsize(7.5)
    for (row, col), cell in t_leakage.get_celld().items():
        if row == 0:
            cell.set_facecolor('#0d3b66')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else 'white')

    ax.text(0.08, 0.47, "2. Selected Defender-Observable Features (8 Features)", fontsize=13, fontweight='bold', color='#0d3b66')
    obs_desc = (
        "Only features that a real-world host-based sensor or network monitor could observe were retained.\n"
        "Both instantaneous telemetry counts and calculated step-to-step deltas were utilized:"
    )
    ax.text(0.08, 0.41, obs_desc, fontsize=9, color='#2b2d42', linespacing=1.3)
    
    features_table = [
        ["Selected Feature", "Telemetry Type", "Correlation with Attack", "Random Forest Importance"],
        ["observation_size_before", "State Byte Size", "+0.1579", "61.55% (Primary Indicator)"],
        ["observation_size_delta", "Step Transition Delta", "+0.0563", "19.60% (Secondary Indicator)"],
        ["session_count_before", "Active Network Sessions", "+0.1388", "7.71%"],
        ["red_session_count_before", "Compromised Sessions", "+0.1388", "5.06%"],
        ["process_delta", "Process Count Change", "+0.0201", "2.84%"],
        ["process_count_before", "Active Running Processes", "+0.1384", "1.71%"],
        ["session_delta", "Session Count Change", "-0.0001", "0.76%"],
        ["red_session_delta", "Red Session Change", "-0.0001", "0.76%"]
    ]
    t_feat = ax.table(cellText=features_table[1:], colLabels=features_table[0], loc='center',
                      cellLoc='left', bbox=[0.08, 0.12, 0.84, 0.27])
    t_feat.auto_set_font_size(False)
    t_feat.set_fontsize(7.5)
    for (row, col), cell in t_feat.get_celld().items():
        if row == 0:
            cell.set_facecolor('#0d3b66')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else 'white')

    ax.text(0.5, 0.04, "Page 2 of 4   |   Project Argus — Phase 2 Technical Report", fontsize=8, ha='center', color='#8d99ae')
    pdf.savefig(fig, dpi=150)
    plt.close()

    # =========================================================================
    # PAGE 3: EXPERIMENTAL RESULTS & CONFUSION MATRIX
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    ax.text(0.5, 0.95, "EXPERIMENTAL RESULTS & DRIFT ANALYSIS", fontsize=18, fontweight='bold', ha='center', color='#0d3b66')
    ax.axhline(y=0.92, xmin=0.08, xmax=0.92, color='#0d3b66', linewidth=2)
    
    ax.text(0.08, 0.88, "1. Disaggregated Performance Across Attacker Adaptation", fontsize=13, fontweight='bold', color='#0d3b66')
    split_desc = (
        "Training Set : Low + Medium Change (1,000 samples | 328 Benign, 672 Attack)\n"
        "Testing Set  : High Change Out-of-Distribution (500 samples | 170 Benign, 330 Attack)"
    )
    ax.text(0.08, 0.82, split_desc, fontsize=8.5, color='#2b2d42', family='monospace')
    
    metrics_table = [
        ["Condition", "Samples", "Accuracy", "Precision", "Recall (Attacks)", "F1-Score", "Attacks Caught", "False Alarms"],
        ["Low Change", "500", "94.40%", "94.79%", "91.00%", "0.9286", "182 / 200 (91.0%)", "10 / 300 ( 3.3%)"],
        ["Medium Change", "500", "82.00%", "97.51%", "83.05%", "0.8970", "392 / 472 (83.1%)", "10 /  28 (35.7%)"],
        ["High Change (Test)", "500", "67.20%", "68.36%", "93.64%", "0.7903", "309 / 330 (93.6%)", "143 / 170 (84.1%) ⚠️"],
        ["Overall Dataset", "1,500", "81.20%", "84.42%", "88.12%", "0.8623", "883 / 1002 (88.1%)", "163 / 498 (32.7%)"]
    ]
    t_metrics = ax.table(cellText=metrics_table[1:], colLabels=metrics_table[0], loc='center',
                         cellLoc='center', bbox=[0.08, 0.63, 0.84, 0.17])
    t_metrics.auto_set_font_size(False)
    t_metrics.set_fontsize(7.5)
    for (row, col), cell in t_metrics.get_celld().items():
        if row == 0:
            cell.set_facecolor('#0d3b66')
            cell.set_text_props(color='white', weight='bold')
        elif row == 3: # Highlight High Change
            cell.set_facecolor('#ffe5d9')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else 'white')
            
    # Embed Confusion Matrix Plot
    ax.text(0.08, 0.58, "2. Confusion Matrix Comparison (Full Dataset vs. High Attacker Drift)", fontsize=13, fontweight='bold', color='#0d3b66')
    if CM_IMAGE.exists():
        img = mpimg.imread(str(CM_IMAGE))
        ax_img = fig.add_axes([0.10, 0.24, 0.80, 0.31])
        ax_img.imshow(img)
        ax_img.axis('off')
        
    cm_caption = (
        "Key Visual Insight: The left matrix demonstrates strong baseline separation on familiar telemetry.\n"
        "The right matrix reveals the catastrophic false-alarm crisis under high attacker change (143 False Positives in orange)."
    )
    ax.text(0.5, 0.18, cm_caption, fontsize=8.5, ha='center', color='#555555', style='italic')

    ax.text(0.5, 0.04, "Page 3 of 4   |   Project Argus — Phase 2 Technical Report", fontsize=8, ha='center', color='#8d99ae')
    pdf.savefig(fig, dpi=150)
    plt.close()

    # =========================================================================
    # PAGE 4: DIAGNOSTIC TAKEAWAYS & ROADMAP TO PHASE 3
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    ax.text(0.5, 0.95, "DIAGNOSTIC ANALYSIS & PHASE 3 ROADMAP", fontsize=18, fontweight='bold', ha='center', color='#0d3b66')
    ax.axhline(y=0.92, xmin=0.08, xmax=0.92, color='#0d3b66', linewidth=2)
    
    ax.text(0.08, 0.88, "1. Deep Diagnostic: Why Did the Static Model Fail?", fontsize=13, fontweight='bold', color='#0d3b66')
    diag_text = (
        "A critical scientific finding from Phase 2 is the exact mechanism of failure under High Attacker Change:\n\n"
        "  1. Attacks Were Still Successfully Detected (High Recall = 93.64%):\n"
        "     The static model did not fail by falling asleep or missing active intrusions. It caught 309 out of 330 attacks.\n\n"
        "  2. Failure Mode: The False Alarm Crisis (Precision Collapsed to 68.36%):\n"
        "     Under High Attacker Change, the baseline operational telemetry of the swarm shifts significantly.\n"
        "     Because the static model's decision boundaries were fixed during Low/Med training, it misinterprets normal\n"
        "     operational drift as hostile actions, generating 143 false alarms out of 170 benign steps (84.1% error rate).\n\n"
        "  3. Operational Consequence in Drone Swarms:\n"
        "     In an autonomous drone network, an 84% false alarm rate would cause drones to unnecessarily sever\n"
        "     communications, reboot processes, or drain batteries—effectively achieving the attacker's denial-of-service goal."
    )
    ax.text(0.08, 0.62, diag_text, fontsize=8.8, color='#2b2d42', linespacing=1.35)
    
    ax.text(0.08, 0.56, "2. Roadmap & Architectural Options for Phase 3", fontsize=13, fontweight='bold', color='#0d3b66')
    p3_options = (
        "Phase 3 requires transitioning from static ML into Adaptive / Incremental Learning to solve the false alarm crisis.\n"
        "Three distinct architectures have been evaluated for implementation:"
    )
    ax.text(0.08, 0.50, p3_options, fontsize=8.8, color='#2b2d42', linespacing=1.3)
    
    options_table = [
        ["Architecture", "Paradigm", "Drift Adaptation Mechanism", "Fit for Drone Swarms"],
        ["Adaptive Random Forest (ARF)", "Streaming Supervised", "Hoeffding Trees + ADWIN drift detectors replace drifted branches", "Top Pick: Native streaming, highly accurate"],
        ["Sliding-Window Retrained RF/XGB", "Windowed Supervised", "Maintains moving buffer (e.g. 300 steps); periodically refits model", "Robust: Simple to explain and implement"],
        ["Streaming Isolation Forest", "Streaming Unsupervised", "Updates tree mass without requiring real-time attack labels", "Strong for zero-day anomaly detection"]
    ]
    t_opt = ax.table(cellText=options_table[1:], colLabels=options_table[0], loc='center',
                     cellLoc='left', bbox=[0.08, 0.24, 0.84, 0.24])
    t_opt.auto_set_font_size(False)
    t_opt.set_fontsize(7.5)
    for (row, col), cell in t_opt.get_celld().items():
        if row == 0:
            cell.set_facecolor('#0d3b66')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else 'white')
            
    ax.text(0.08, 0.18, "3. Transition Milestone to Phase 3", fontsize=13, fontweight='bold', color='#0d3b66')
    trans_text = (
        "Phase 2 stands fully completed, verified, and documented. The baseline metrics establish the definitive\n"
        "empirical benchmark against which all adaptive learning architectures in Phase 3 will be measured."
    )
    ax.text(0.08, 0.11, trans_text, fontsize=8.8, color='#2b2d42', linespacing=1.35)

    ax.text(0.5, 0.04, "Page 4 of 4   |   Project Argus — Phase 2 Technical Report", fontsize=8, ha='center', color='#8d99ae')
    pdf.savefig(fig, dpi=150)
    plt.close()

print(f"[SUCCESS] High-quality 4-Page PDF generated at: {OUTPUT_PDF.resolve()}")
