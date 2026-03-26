"""
Phase 2 — Topological Feature Extraction
=========================================
Loads phase1-dataset.pkl, normalises each hidden-state row to unit length,
builds a Vietoris-Rips filtration with Gudhi, computes H0/H1 persistence
features per window, and saves them to phase2-features.csv.

Produces:
  phase2-persistence-diagrams.png  — birth/death scatter for one looping & one normal window
  phase2-barcodes.png              — barcode plots for the same two windows
  phase2-feature-distributions.png — violin + strip plots of the four H1 features
  phase2-correlations.png          — feature correlation heatmap
  phase2-features.csv              — one row per window with all five features
"""

import pickle
import warnings

import gudhi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── plot style ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")

# ── Gudhi / filtration settings ──────────────────────────────────────────────
MAX_EDGE_LENGTH = 2.0
MAX_DIMENSION = 2
MIN_LIFETIME = 0.01      # bars with lifetime ≤ this are ignored for count/entropy
H0_EVAL_VALUE = 1.0      # filtration value at which to count alive H0 components


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalise each row of *matrix* to unit L2 length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
    return matrix / norms


def compute_tda_features(hidden_matrix: np.ndarray) -> dict:
    """
    Compute H0/H1 persistence features for a single (10 × 768) window.

    Steps
    -----
    1. Normalise rows to unit length.
    2. Build a Vietoris-Rips complex (max_edge_length=2, max_dim=2).
    3. Compute persistence.
    4. Extract H1 bars and five scalar features.

    Returns a dict with keys:
        total_h1, max_h1, count_h1, entropy_h1, h0_components
    """
    pts = normalise_rows(hidden_matrix)

    rips = gudhi.RipsComplex(points=pts, max_edge_length=MAX_EDGE_LENGTH)
    st = rips.create_simplex_tree(max_dimension=MAX_DIMENSION)
    st.compute_persistence()

    pairs = st.persistence()   # list of (dim, (birth, death))

    # ── H1 features ──────────────────────────────────────────────────────────
    h1_lifetimes = [
        d - b for dim, (b, d) in pairs
        if dim == 1 and not np.isinf(d)
    ]

    if h1_lifetimes:
        total_h1 = float(np.sum(h1_lifetimes))
        max_h1 = float(np.max(h1_lifetimes))
        count_h1 = int(np.sum(np.array(h1_lifetimes) > MIN_LIFETIME))
        # Shannon entropy of the lifetime distribution (add tiny epsilon to
        # avoid log(0) when all mass is on a single bar)
        probs = np.array(h1_lifetimes) / (np.sum(h1_lifetimes) + 1e-12)
        entropy_h1 = float(scipy_entropy(probs + 1e-12))
    else:
        total_h1 = 0.0
        max_h1 = 0.0
        count_h1 = 0
        entropy_h1 = 0.0

    # ── H0 control feature: components alive at filtration value 1.0 ─────────
    h0_components = sum(
        1 for dim, (b, d) in pairs
        if dim == 0 and b <= H0_EVAL_VALUE <= (d if not np.isinf(d) else float("inf"))
    )

    return {
        "total_h1": total_h1,
        "max_h1": max_h1,
        "count_h1": count_h1,
        "entropy_h1": entropy_h1,
        "h0_components": h0_components,
    }


def get_persistence_by_dim(hidden_matrix: np.ndarray) -> dict[int, list[tuple[float, float]]]:
    """Return persistence pairs grouped by dimension {0: [...], 1: [...]}."""
    pts = normalise_rows(hidden_matrix)
    rips = gudhi.RipsComplex(points=pts, max_edge_length=MAX_EDGE_LENGTH)
    st = rips.create_simplex_tree(max_dimension=MAX_DIMENSION)
    st.compute_persistence()
    result: dict[int, list] = {0: [], 1: []}
    for dim, (b, d) in st.persistence():
        if dim in result and not np.isinf(d):
            result[dim].append((b, d))
    return result


def main():
    warnings.filterwarnings("ignore")

    # ── load phase 1 data ─────────────────────────────────────────────────────
    with open("phase1-dataset.pkl", "rb") as f:
        windows = pickle.load(f)

    print(f"Loaded {len(windows)} windows from phase1-dataset.pkl")

    rows: list[dict] = []
    for w in tqdm(windows, desc="Computing TDA features"):
        feats = compute_tda_features(w["hidden_state_matrix"])
        rows.append(
            {
                "prompt_id": w["prompt_id"],
                "step_index": w["step_index"],
                "loop_label": w["loop_label"],
                **feats,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv("phase2-features.csv", index=False)
    print("Saved phase2-features.csv")

    # ── grouped statistics ────────────────────────────────────────────────────
    print("\n=== Feature means/stds by loop_label ===")
    feat_cols = ["total_h1", "max_h1", "count_h1", "entropy_h1", "h0_components"]
    print(df.groupby("loop_label")[feat_cols].agg(["mean", "std"]).to_string())

    # ── gate check ────────────────────────────────────────────────────────────
    grouped_means = df.groupby("loop_label")["total_h1"].mean()
    if len(grouped_means) >= 2:
        diff = abs(grouped_means.iloc[1] - grouped_means.iloc[0])
    else:
        diff = 0.0

    print(f"\ntotal_h1 means: {grouped_means.to_dict()}")
    if diff > 0.05:
        print("GATE PASSED")
    else:
        print(
            f"GATE FAILED — STOP HERE\n"
            f"Difference in mean total_h1 = {diff:.4f} (need > 0.05).\n"
            f"Means: {grouped_means.to_dict()}"
        )

    # =========================================================================
    # Pick representative windows for diagram / barcode plots
    # =========================================================================
    loop_idx = df[df["loop_label"] == 1].index.tolist()
    norm_idx = df[df["loop_label"] == 0].index.tolist()

    rep_loop_idx = loop_idx[0] if loop_idx else 0
    rep_norm_idx = norm_idx[0] if norm_idx else 1

    rep_loop_pairs = get_persistence_by_dim(windows[rep_loop_idx]["hidden_state_matrix"])
    rep_norm_pairs = get_persistence_by_dim(windows[rep_norm_idx]["hidden_state_matrix"])

    labels_map = {0: "Looping (label=1)", 1: "Normal (label=0)"}
    pairs_list = [rep_loop_pairs, rep_norm_pairs]
    titles = ["Looping Window", "Normal Window"]

    # =========================================================================
    # PLOT 1 — Persistence diagrams
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pairs, title in zip(axes, pairs_list, titles):
        diag_max = 0.0
        for dim, color in [(0, "blue"), (1, "red")]:
            pts_d = pairs.get(dim, [])
            if pts_d:
                xs, ys = zip(*pts_d)
                ax.scatter(xs, ys, c=color, label=f"H{dim}", alpha=0.7, s=30)
                diag_max = max(diag_max, max(ys))
        ax.plot([0, diag_max + 0.1], [0, diag_max + 0.1], "k--", linewidth=1, label="Diagonal")
        ax.set_title(f"Persistence Diagram\n{title}")
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("phase2-persistence-diagrams.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 2 — Persistence barcodes
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dim_colors = {0: "steelblue", 1: "tomato"}

    for ax, pairs, title in zip(axes, pairs_list, titles):
        y = 0
        handles = {}
        for dim in [0, 1]:
            color = dim_colors[dim]
            for b, d in pairs.get(dim, []):
                h = ax.barh(y, d - b, left=b, height=0.6, color=color, alpha=0.8)
                if dim not in handles:
                    handles[dim] = h
                y += 1
        ax.set_title(f"Persistence Barcode\n{title}")
        ax.set_xlabel("Filtration Value")
        ax.set_ylabel("Bar Index")
        ax.legend(
            [handles[d] for d in sorted(handles)],
            [f"H{d}" for d in sorted(handles)],
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig("phase2-barcodes.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 3 — Feature distributions (violin + strip)
    # =========================================================================
    h1_features = ["total_h1", "max_h1", "count_h1", "entropy_h1"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, feat in zip(axes.flat, h1_features):
        sns.violinplot(
            data=df,
            x="loop_label",
            y=feat,
            ax=ax,
            palette="Set2",
            inner=None,
            cut=0,
        )
        sns.stripplot(
            data=df,
            x="loop_label",
            y=feat,
            ax=ax,
            color="black",
            alpha=0.3,
            size=2,
            jitter=True,
        )
        ax.set_title(f"Distribution of {feat}")
        ax.set_xlabel("Loop Label (0=Normal, 1=Looping)")
        ax.set_ylabel(feat)

    plt.suptitle("Phase 2 — H1 Feature Distributions by Loop Label", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("phase2-feature-distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT 4 — Feature correlation heatmap
    # =========================================================================
    corr_cols = feat_cols + ["loop_label"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )
    ax.set_title("Phase 2 — Feature Correlation Matrix (incl. loop_label)")
    plt.tight_layout()
    plt.savefig("phase2-correlations.png", dpi=150)
    plt.close()

    # ── done ──────────────────────────────────────────────────────────────────
    print("\n=== PHASE COMPLETE ===")
    print("Files saved:")
    print("  phase2-features.csv")
    print("  phase2-persistence-diagrams.png")
    print("  phase2-barcodes.png")
    print("  phase2-feature-distributions.png")
    print("  phase2-correlations.png")


if __name__ == "__main__":
    main()
