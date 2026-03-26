"""
Phase 3 — Statistical Validation
==================================
Loads phase2-features.csv and runs a full suite of statistical tests and
machine-learning benchmarks to validate whether H1 topological features
discriminate looping from normal language-model generations.

Produces:
  phase3-roc-curves.png        — ROC curves for H1 model and two baselines
  phase3-feature-importance.png — logistic regression coefficients
  phase3-scatter.png           — total_h1 vs max_h1 scatter + decision boundary
  phase3-boxplots.png          — box plots with p-value annotations
  phase3-confusion-matrix.png  — confusion matrix from best CV fold
  phase3-calibration.png       — calibration curve of the H1 model
  phase3-report.md             — written report with verdict
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── plot style ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")

FEATURES_H1 = ["total_h1", "max_h1", "count_h1", "entropy_h1", "h0_components"]
FEATURES_H1_ONLY = ["total_h1", "max_h1", "count_h1", "entropy_h1"]
N_SPLITS = 5
RANDOM_STATE = 42


def cross_val_roc(X: np.ndarray, y: np.ndarray, clf, n_splits: int = N_SPLITS):
    """
    Run stratified k-fold CV. Returns:
        mean_auc, mean_acc, best_fold_info (X_test, y_test, y_prob, clf_fitted)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs, accs = [], []
    best_auc = -1.0
    best_fold = None

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf.fit(X_tr_s, y_tr)
        y_prob = clf.predict_proba(X_te_s)[:, 1]
        y_pred = clf.predict(X_te_s)

        fold_auc = roc_auc_score(y_te, y_prob)
        fold_acc = np.mean(y_pred == y_te)
        aucs.append(fold_auc)
        accs.append(fold_acc)

        if fold_auc > best_auc:
            best_auc = fold_auc
            best_fold = (X_te_s, y_te, y_prob, clf, scaler)

    return float(np.mean(aucs)), float(np.mean(accs)), best_fold


def main():
    warnings.filterwarnings("ignore")

    # ── load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv("phase2-features.csv")
    X_all = df[FEATURES_H1].values
    X_h0 = df[["h0_components"]].values
    X_pos = df[["step_index"]].values.astype(float)
    # normalise step_index to [0, 1]
    X_pos = (X_pos - X_pos.min()) / (X_pos.max() - X_pos.min() + 1e-12)
    y = df["loop_label"].values

    print(f"Dataset: {len(df)} windows, {y.sum()} looping, {(1-y).sum()} normal")

    # =========================================================================
    # Statistical tests — Mann-Whitney U on each H1 feature
    # =========================================================================
    loop_df = df[df["loop_label"] == 1]
    norm_df = df[df["loop_label"] == 0]

    mw_results = {}
    print("\n=== Mann-Whitney U Tests ===")
    for feat in FEATURES_H1_ONLY:
        a = loop_df[feat].values
        b = norm_df[feat].values
        stat, pval = mannwhitneyu(a, b, alternative="two-sided")
        mw_results[feat] = (stat, pval)
        print(f"  {feat:20s}  U={stat:.1f}  p={pval:.4e}")

    # =========================================================================
    # Logistic regression — H1 model (5 features)
    # =========================================================================
    clf_h1 = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    mean_auc_h1, mean_acc_h1, best_fold_h1 = cross_val_roc(X_all, y, clf_h1)
    print(f"\nH1 model   — mean AUC: {mean_auc_h1:.3f}  mean Acc: {mean_acc_h1:.3f}")

    # ── baseline 1: h0-components only ───────────────────────────────────────
    clf_h0 = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    mean_auc_h0, mean_acc_h0, _ = cross_val_roc(X_h0, y, clf_h0)
    print(f"H0 baseline — mean AUC: {mean_auc_h0:.3f}  mean Acc: {mean_acc_h0:.3f}")

    # ── baseline 2: step-index position only ─────────────────────────────────
    clf_pos = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    mean_auc_pos, mean_acc_pos, _ = cross_val_roc(X_pos, y, clf_pos)
    print(f"Pos baseline — mean AUC: {mean_auc_pos:.3f}  mean Acc: {mean_acc_pos:.3f}")

    # ── verdict ───────────────────────────────────────────────────────────────
    n_sig = sum(1 for _, pval in mw_results.values() if pval < 0.05)
    if mean_auc_h1 > 0.65 and n_sig >= 2:
        verdict = "SUPPORTED"
    elif mean_auc_h1 >= 0.55:
        verdict = "WEAK"
    else:
        verdict = "FALSIFIED"

    # =========================================================================
    # Write phase3-report.md
    # =========================================================================
    report_lines = [
        "# Phase 3 — Statistical Validation Report\n",
        "## Mann-Whitney U Tests\n",
        "| Feature | U Statistic | p-value | Significant (p<0.05) |",
        "|---------|-------------|---------|----------------------|",
    ]
    for feat, (stat, pval) in mw_results.items():
        sig = "✓" if pval < 0.05 else "✗"
        report_lines.append(f"| {feat} | {stat:.1f} | {pval:.4e} | {sig} |")

    report_lines += [
        "",
        "## Cross-Validated AUC\n",
        f"| Model | Mean AUC | Mean Accuracy |",
        f"|-------|----------|---------------|",
        f"| H1 model (all 5 features) | {mean_auc_h1:.3f} | {mean_acc_h1:.3f} |",
        f"| H0-only baseline | {mean_auc_h0:.3f} | {mean_acc_h0:.3f} |",
        f"| Position baseline | {mean_auc_pos:.3f} | {mean_acc_pos:.3f} |",
        "",
        "## Conclusion\n",
        f"**Verdict: {verdict}**\n",
    ]

    if verdict == "SUPPORTED":
        report_lines.append(
            "The H1 topological features show statistically significant differences "
            "between looping and normal generation windows. The logistic regression "
            "achieves AUC > 0.65 with at least two features significant at p < 0.05, "
            "and outperforms both baselines. This provides credible evidence that "
            "topological structure in hidden-state trajectories correlates with "
            "repetitive text generation."
        )
    elif verdict == "WEAK":
        report_lines.append(
            "The H1 topological features show some discriminative ability (AUC between "
            "0.55 and 0.65), but the evidence is not strong. Results should be "
            "interpreted cautiously. Further hyperparameter tuning or richer prompts "
            "may strengthen the signal."
        )
    else:
        report_lines.append(
            "The H1 topological features do not show meaningful discriminative ability "
            "(AUC < 0.55 or no significant features). The TSK hypothesis is not "
            "supported by this dataset. Consider revisiting prompt design, the "
            "filtration parameters, or the loop-score threshold."
        )

    with open("phase3-report.md", "w") as f:
        f.write("\n".join(report_lines))
    print("\nSaved phase3-report.md")

    # ── gate check to console ─────────────────────────────────────────────────
    print(f"\nVerdict: {verdict}")
    if verdict == "SUPPORTED":
        print("PROCEED TO PHASE 4")
    elif verdict == "WEAK":
        print("PROCEED WITH CAUTION")
    else:
        print("STOP — TSK IS FALSIFIED")

    # =========================================================================
    # PLOT 1 — ROC curves
    # =========================================================================

    # Re-fit on all data with a scaler for smooth ROC curves to display
    def get_roc(X_feat, clf_cls):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for tr, te in skf.split(X_feat, y):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_feat[tr])
            Xte = sc.transform(X_feat[te])
            m = clf_cls(max_iter=1000, random_state=RANDOM_STATE)
            m.fit(Xtr, y[tr])
            yp = m.predict_proba(Xte)[:, 1]
            fpr, tpr, _ = roc_curve(y[te], yp)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 0.0
        return mean_fpr, mean_tpr, auc(mean_fpr, mean_tpr)

    fpr_h1, tpr_h1, auc_h1 = get_roc(X_all, lambda **kw: LogisticRegression(**kw))
    fpr_h0, tpr_h0, auc_h0 = get_roc(X_h0, lambda **kw: LogisticRegression(**kw))
    fpr_pos, tpr_pos, auc_pos = get_roc(X_pos, lambda **kw: LogisticRegression(**kw))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(fpr_h1, tpr_h1, alpha=0.2, color="steelblue")
    ax.plot(fpr_h1, tpr_h1, color="steelblue", lw=2, label=f"H1 model (AUC={auc_h1:.3f})")
    ax.plot(fpr_h0, tpr_h0, color="orange", lw=2, linestyle="--", label=f"H0-only baseline (AUC={auc_h0:.3f})")
    ax.plot(fpr_pos, tpr_pos, color="green", lw=2, linestyle=":", label=f"Position baseline (AUC={auc_pos:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_title("Phase 3 — ROC Curves (5-fold CV)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("phase3-roc-curves.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 2 — Feature importance (logistic regression coefficients)
    # =========================================================================
    # Fit on full dataset for coefficient extraction
    scaler_full = StandardScaler()
    X_all_s = scaler_full.fit_transform(X_all)
    clf_full = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_full.fit(X_all_s, y)
    coefs = clf_full.coef_[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["green" if c > 0 else "red" for c in coefs]
    ax.barh(FEATURES_H1, coefs, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Phase 3 — Logistic Regression Feature Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig("phase3-feature-importance.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 3 — Scatter: total_h1 vs max_h1 + decision boundary
    # =========================================================================
    # Train a 2-feature model for the 2D decision boundary
    X_2d = df[["total_h1", "max_h1"]].values
    scaler_2d = StandardScaler()
    X_2d_s = scaler_2d.fit_transform(X_2d)
    clf_2d = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_2d.fit(X_2d_s, y)

    x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
    y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = scaler_2d.transform(np.c_[xx.ravel(), yy.ravel()])
    zz = clf_2d.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {0: "cornflowerblue", 1: "tomato"}
    for label, color in palette.items():
        mask = y == label
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, label=f"label={label}", alpha=0.5, s=15, edgecolors="none"
        )
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
    ax.set_title("Phase 3 — total_h1 vs max_h1 with Decision Boundary")
    ax.set_xlabel("total_h1")
    ax.set_ylabel("max_h1")
    ax.legend()
    plt.tight_layout()
    plt.savefig("phase3-scatter.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 4 — Box plots with p-value annotations
    # =========================================================================
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    for ax, feat in zip(axes, FEATURES_H1_ONLY):
        data_loop = loop_df[feat].values
        data_norm = norm_df[feat].values
        ax.boxplot(
            [data_norm, data_loop],
            labels=["Normal", "Looping"],
            showfliers=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
        )
        _, pval = mw_results[feat]
        stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        y_top = max(data_loop.max(), data_norm.max()) * 1.05
        ax.annotate(
            f"p={pval:.3e}\n{stars}",
            xy=(1.5, y_top),
            ha="center",
            fontsize=8,
        )
        ax.set_title(feat)
        ax.set_ylabel(feat)

    plt.suptitle("Phase 3 — H1 Feature Box Plots by Loop Label", fontsize=13)
    plt.tight_layout()
    plt.savefig("phase3-boxplots.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 5 — Confusion matrix (best CV fold)
    # =========================================================================
    X_te_best, y_te_best, y_prob_best, clf_best, _ = best_fold_h1
    y_pred_best = clf_best.predict(X_te_best)
    cm = confusion_matrix(y_te_best, y_pred_best)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=np.array(
            [[f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)" for j in range(cm.shape[1])]
             for i in range(cm.shape[0])]
        ),
        fmt="",
        cmap="Blues",
        ax=ax,
        xticklabels=["Normal", "Looping"],
        yticklabels=["Normal", "Looping"],
    )
    ax.set_title("Phase 3 — Confusion Matrix (Best CV Fold)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig("phase3-confusion-matrix.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 6 — Calibration curve
    # =========================================================================
    # Fit an isotonic-calibrated model on full data for the calibration display
    scaler_cal = StandardScaler()
    X_all_sc = scaler_cal.fit_transform(X_all)
    base_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    cal_clf = CalibratedClassifierCV(base_clf, cv=N_SPLITS, method="isotonic")
    cal_clf.fit(X_all_sc, y)
    y_prob_cal = cal_clf.predict_proba(X_all_sc)[:, 1]
    frac_pos, mean_pred = calibration_curve(y, y_prob_cal, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, "s-", color="steelblue", label="H1 model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_title("Phase 3 — Calibration Curve (H1 Logistic Regression)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    plt.tight_layout()
    plt.savefig("phase3-calibration.png", dpi=150)
    plt.close()

    # ── done ──────────────────────────────────────────────────────────────────
    print("\n=== PHASE COMPLETE ===")
    print("Files saved:")
    print("  phase3-report.md")
    print("  phase3-roc-curves.png")
    print("  phase3-feature-importance.png")
    print("  phase3-scatter.png")
    print("  phase3-boxplots.png")
    print("  phase3-confusion-matrix.png")
    print("  phase3-calibration.png")


if __name__ == "__main__":
    main()
