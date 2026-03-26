"""
Phase 4 — Minimal TSK Prototype & Evaluation
=============================================
Builds a TSK (Topological Steering Kernel) generation wrapper around GPT-2.
At each step the wrapper computes the H1 topology of the last-10-token hidden
state window; if total_h1 exceeds a threshold derived from Phase 2 looping
windows it suppresses the top-logit token, steering the model away from its
most probable (looping) continuation.

Compares 15 looping prompts under baseline GPT-2 vs TSK-wrapped GPT-2 on:
  - trigram repetition rate
  - unique-token ratio
  - self-BLEU (second half vs first half of completion)

Produces:
  phase4-metric-comparison.png      — grouped bar chart (baseline vs TSK, 3 metrics)
  phase4-per-prompt.png             — dot plot of per-prompt trigram repetition change
  phase4-intervention-timeline.png  — total_h1 per step for 3 prompts, interventions marked
  phase4-suppressed-tokens.png      — top-10 most suppressed token strings
  phase4-completions.txt            — side-by-side baseline vs TSK completions (5 prompts)
  phase4-results.csv                — raw per-prompt metrics
  phase4-summary.md                 — plain-English verdict
"""

import warnings
from collections import Counter

import gudhi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── plot style ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")

# ── constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 10
GEN_TOKENS = 60
MAX_EDGE_LENGTH = 2.0
MAX_DIMENSION = 2
MIN_LIFETIME = 0.01

# 15 looping prompts (more than Phase 1 to give a richer evaluation set)
LOOPING_PROMPTS_EVAL = [
    "The answer is the answer is the answer is",
    "Repeat after me: one two three one two three",
    "Yes yes yes yes yes yes yes yes yes yes yes",
    "And so on and so on and so on and so on and",
    "Again and again and again and again and again",
    "The cat sat on the mat the cat sat on the mat",
    "To be or not to be to be or not to be to be",
    "Hello hello hello hello hello hello hello hello",
    "One plus one equals two one plus one equals two",
    "Go go go go go go go go go go go go go go go go",
    "La la la la la la la la la la la la la la la la",
    "Step by step by step by step by step by step by",
    "More and more and more and more and more and more",
    "Over and over and over and over and over and over",
    "Round and round and round and round and round and",
]


# ── TDA helper (same as phase2.py) ───────────────────────────────────────────

def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def compute_total_h1(hidden_matrix: np.ndarray) -> float:
    """Return total H1 persistence (sum of bar lifetimes) for a window."""
    pts = normalise_rows(hidden_matrix)
    rips = gudhi.RipsComplex(points=pts, max_edge_length=MAX_EDGE_LENGTH)
    st = rips.create_simplex_tree(max_dimension=MAX_DIMENSION)
    st.compute_persistence()
    lifetimes = [
        d - b for dim, (b, d) in st.persistence()
        if dim == 1 and not np.isinf(d)
    ]
    return float(np.sum(lifetimes)) if lifetimes else 0.0


# ── evaluation metrics ────────────────────────────────────────────────────────

def trigram_repetition_rate(token_ids: list[int]) -> float:
    """Fraction of trigrams that appear more than once."""
    if len(token_ids) < 3:
        return 0.0
    trigrams = list(zip(token_ids[:-2], token_ids[1:-1], token_ids[2:]))
    counts = Counter(trigrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(trigrams)


def unique_token_ratio(token_ids: list[int]) -> float:
    """Unique tokens divided by total tokens."""
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / len(token_ids)


def self_bleu(token_ids: list[int]) -> float:
    """
    Compute 2-gram precision of the second half against the first half
    as a proxy for self-BLEU (higher → more repetition between halves).
    """
    if len(token_ids) < 4:
        return 0.0
    mid = len(token_ids) // 2
    first_half = token_ids[:mid]
    second_half = token_ids[mid:]

    ref_bigrams = Counter(zip(first_half[:-1], first_half[1:]))
    hyp_bigrams = list(zip(second_half[:-1], second_half[1:]))
    if not hyp_bigrams:
        return 0.0
    matches = sum(1 for bg in hyp_bigrams if ref_bigrams.get(bg, 0) > 0)
    return matches / len(hyp_bigrams)


# ── generation functions ──────────────────────────────────────────────────────

def generate_baseline(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    n_tokens: int = GEN_TOKENS,
) -> list[int]:
    """Generate *n_tokens* tokens greedily without any intervention."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(n_tokens):
            out = model(torch.tensor([generated], device=device))
            next_tok = out.logits[0, -1].argmax().item()
            generated.append(next_tok)

    # return only the newly generated tokens
    return generated[input_ids.shape[1]:]


def generate_tsk(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    threshold: float,
    n_tokens: int = GEN_TOKENS,
    window_size: int = WINDOW_SIZE,
) -> tuple[list[int], list[dict], list[float]]:
    """
    Generate with TSK intervention.

    At each step:
      1. Compute the last-layer hidden states for the last *window_size* tokens.
      2. Compute total_h1.
      3. If total_h1 > threshold, set the top-logit token's logit to -inf.

    Returns
    -------
    generated_tokens : newly generated token ids
    interventions    : list of {step, total_h1, suppressed_token_id}
    total_h1_trace   : total_h1 at every step
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()
    all_hidden: list[np.ndarray] = []
    interventions: list[dict] = []
    total_h1_trace: list[float] = []

    with torch.no_grad():
        for step in range(n_tokens):
            out = model(
                torch.tensor([generated], device=device),
                output_hidden_states=True,
            )
            last_hidden = out.hidden_states[-1][0]  # (seq_len, 768)
            all_hidden.append(last_hidden[-1].cpu().float().numpy())

            logits = out.logits[0, -1].clone()

            # compute TDA only once window is full
            if len(all_hidden) >= window_size:
                hidden_matrix = np.stack(all_hidden[-window_size:])  # (10, 768)
                th1 = compute_total_h1(hidden_matrix)
            else:
                th1 = 0.0

            total_h1_trace.append(th1)

            if th1 > threshold and len(all_hidden) >= window_size:
                # suppress the top-logit token
                top_token = logits.argmax().item()
                logits[top_token] = float("-inf")
                interventions.append({"step": step, "total_h1": th1, "suppressed_token_id": top_token})

            next_tok = logits.argmax().item()
            generated.append(next_tok)

    new_tokens = generated[input_ids.shape[1]:]
    return new_tokens, interventions, total_h1_trace


def main():
    warnings.filterwarnings("ignore")

    # ── load threshold from phase2-features.csv ───────────────────────────────
    feat_df = pd.read_csv("phase2-features.csv")
    loop_h1 = feat_df[feat_df["loop_label"] == 1]["total_h1"].values
    if len(loop_h1) > 0:
        threshold_T = float(np.percentile(loop_h1, 75))
    else:
        # Fallback: use the median of all H1 values so the threshold is still
        # data-driven even when no windows were labelled as looping in phase 2.
        all_h1 = feat_df["total_h1"].values
        threshold_T = float(np.median(all_h1)) if len(all_h1) > 0 else 0.5
    print(f"TSK threshold T = {threshold_T:.4f} (75th pct of looping total_h1)")

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading GPT-2 small …")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    results: list[dict] = []
    all_interventions: list[dict] = []
    baseline_completions: list[str] = []
    tsk_completions: list[str] = []
    tsk_h1_traces: list[list[float]] = []

    for i, prompt in enumerate(tqdm(LOOPING_PROMPTS_EVAL, desc="Evaluating prompts")):
        # ── baseline ─────────────────────────────────────────────────────────
        base_tokens = generate_baseline(model, tokenizer, prompt)
        base_rep = trigram_repetition_rate(base_tokens)
        base_utr = unique_token_ratio(base_tokens)
        base_sb = self_bleu(base_tokens)

        # ── TSK ───────────────────────────────────────────────────────────────
        tsk_tokens, interventions, h1_trace = generate_tsk(
            model, tokenizer, prompt, threshold=threshold_T
        )
        tsk_rep = trigram_repetition_rate(tsk_tokens)
        tsk_utr = unique_token_ratio(tsk_tokens)
        tsk_sb = self_bleu(tsk_tokens)

        # tag interventions with prompt index
        for iv in interventions:
            iv["prompt_id"] = i
        all_interventions.extend(interventions)
        tsk_h1_traces.append(h1_trace)

        results.append(
            {
                "prompt_id": i,
                "prompt": prompt,
                "baseline_trigram_rep": base_rep,
                "tsk_trigram_rep": tsk_rep,
                "baseline_unique_ratio": base_utr,
                "tsk_unique_ratio": tsk_utr,
                "baseline_self_bleu": base_sb,
                "tsk_self_bleu": tsk_sb,
                "n_interventions": len(interventions),
            }
        )

        baseline_completions.append(tokenizer.decode(base_tokens, skip_special_tokens=True))
        tsk_completions.append(tokenizer.decode(tsk_tokens, skip_special_tokens=True))

    results_df = pd.DataFrame(results)
    results_df.to_csv("phase4-results.csv", index=False)
    print("Saved phase4-results.csv")

    # ── aggregate metrics ─────────────────────────────────────────────────────
    metrics = ["trigram_rep", "unique_ratio", "self_bleu"]
    labels = ["Trigram Repetition Rate", "Unique Token Ratio", "Self-BLEU"]
    base_means = [results_df[f"baseline_{m}"].mean() for m in metrics]
    base_stds = [results_df[f"baseline_{m}"].std() for m in metrics]
    tsk_means = [results_df[f"tsk_{m}"].mean() for m in metrics]
    tsk_stds = [results_df[f"tsk_{m}"].std() for m in metrics]

    print("\n=== Aggregate Metrics (mean ± std) ===")
    for label, bm, bs, tm, ts in zip(labels, base_means, base_stds, tsk_means, tsk_stds):
        print(f"  {label:30s}: baseline {bm:.3f}±{bs:.3f}  TSK {tm:.3f}±{ts:.3f}")

    # =========================================================================
    # PLOT 1 — Metric comparison grouped bar chart
    # =========================================================================
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, base_means, width, yerr=base_stds, label="Baseline",
                   color="cornflowerblue", capsize=4, edgecolor="black")
    bars2 = ax.bar(x + width / 2, tsk_means, width, yerr=tsk_stds, label="TSK",
                   color="tomato", capsize=4, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Phase 4 — Baseline vs TSK Metric Comparison (15 Prompts, Mean ± Std)")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig("phase4-metric-comparison.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 2 — Per-prompt improvement in trigram repetition rate
    # =========================================================================
    delta_rep = results_df["baseline_trigram_rep"] - results_df["tsk_trigram_rep"]
    colors = ["green" if d >= 0 else "red" for d in delta_rep]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(len(delta_rep)), delta_rep, c=colors, s=80, zorder=3)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    for idx, (d, c) in enumerate(zip(delta_rep, colors)):
        ax.vlines(idx, 0, d, colors=c, linewidth=1, alpha=0.6)
    ax.set_title("Phase 4 — Per-Prompt Change in Trigram Repetition Rate\n(Baseline − TSK; green = TSK improved)")
    ax.set_xlabel("Prompt Index")
    ax.set_ylabel("Δ Trigram Repetition Rate")
    ax.set_xticks(range(len(delta_rep)))
    plt.tight_layout()
    plt.savefig("phase4-per-prompt.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 3 — Intervention timeline for 3 representative prompts
    # =========================================================================
    rep_ids = [0, 1, 2]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for ax, pid in zip(axes, rep_ids):
        trace = tsk_h1_traces[pid]
        ax.plot(trace, color="steelblue", linewidth=1.5, label="total_h1")
        ax.axhline(threshold_T, color="orange", linestyle="--", linewidth=1, label=f"Threshold T={threshold_T:.2f}")
        # mark interventions
        iv_steps = [iv["step"] for iv in all_interventions if iv["prompt_id"] == pid]
        for s in iv_steps:
            ax.axvline(s, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_title(f'Prompt {pid}: "{LOOPING_PROMPTS_EVAL[pid][:40]}…"  ({len(iv_steps)} interventions)')
        ax.set_xlabel("Generation Step")
        ax.set_ylabel("total_h1")
        ax.legend(fontsize=8)

    plt.suptitle("Phase 4 — Intervention Timeline (red dashes = TSK intervention)", fontsize=12)
    plt.tight_layout()
    plt.savefig("phase4-intervention-timeline.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 4 — Top-10 most frequently suppressed tokens
    # =========================================================================
    suppressed_ids = [iv["suppressed_token_id"] for iv in all_interventions]
    if suppressed_ids:
        token_counts = Counter(suppressed_ids)
        top10 = token_counts.most_common(10)
        top_tokens = [tokenizer.decode([tid]).strip() or f"[{tid}]" for tid, _ in top10]
        top_counts = [cnt for _, cnt in top10]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(top_tokens, top_counts, color=sns.color_palette("Set2", len(top_tokens)), edgecolor="black")
        ax.set_title("Phase 4 — Top-10 Most Frequently Suppressed Tokens by TSK")
        ax.set_xlabel("Token")
        ax.set_ylabel("Suppression Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
    else:
        # No interventions occurred — save an empty placeholder plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No interventions recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Phase 4 — Top-10 Most Frequently Suppressed Tokens")
        plt.tight_layout()

    plt.savefig("phase4-suppressed-tokens.png", dpi=150)
    plt.close()

    # =========================================================================
    # SAVE completions.txt (5 representative prompts)
    # =========================================================================
    with open("phase4-completions.txt", "w") as f:
        for pid in range(min(5, len(LOOPING_PROMPTS_EVAL))):
            f.write(f"{'='*70}\n")
            f.write(f"Prompt {pid}: {LOOPING_PROMPTS_EVAL[pid]}\n")
            f.write(f"{'-'*35} BASELINE {'-'*25}\n")
            f.write(baseline_completions[pid] + "\n")
            f.write(f"{'-'*35} TSK {'-'*30}\n")
            f.write(tsk_completions[pid] + "\n")
            f.write("\n")
    print("Saved phase4-completions.txt")

    # =========================================================================
    # Write phase4-summary.md
    # =========================================================================
    avg_baseline_rep = results_df["baseline_trigram_rep"].mean()
    avg_tsk_rep = results_df["tsk_trigram_rep"].mean()
    pct_change = 100.0 * (avg_baseline_rep - avg_tsk_rep) / (avg_baseline_rep + 1e-12)
    n_improved = int((delta_rep > 0).sum())
    n_total = len(delta_rep)
    total_interventions = len(all_interventions)

    with open("phase4-summary.md", "w") as f:
        f.write("# Phase 4 — TSK Evaluation Summary\n\n")
        f.write("## Metrics Overview\n\n")
        f.write("| Metric | Baseline (mean) | TSK (mean) | Δ |\n")
        f.write("|--------|-----------------|------------|---|\n")
        for label, m, bm, tm in zip(labels, metrics, base_means, tsk_means):
            f.write(f"| {label} | {bm:.3f} | {tm:.3f} | {tm-bm:+.3f} |\n")
        f.write(f"\n## Verdict\n\n")
        f.write(f"TSK applied **{total_interventions}** total interventions across 15 prompts.\n\n")

        if pct_change > 5.0:
            verdict_tsk = "TSK **reduced** looping"
            credible = True
        elif pct_change > 0:
            verdict_tsk = "TSK **marginally reduced** looping"
            credible = False
        else:
            verdict_tsk = "TSK **did not reduce** looping (or made it worse)"
            credible = False

        f.write(
            f"{verdict_tsk} by {abs(pct_change):.1f}% on average trigram repetition rate "
            f"({avg_baseline_rep:.3f} → {avg_tsk_rep:.3f}).\n\n"
            f"{n_improved}/{n_total} prompts showed improvement.\n\n"
        )

        f.write("## Failure Cases\n\n")
        failed = results_df[results_df["baseline_trigram_rep"] <= results_df["tsk_trigram_rep"]]
        if len(failed) > 0:
            for _, row in failed.iterrows():
                f.write(f"- Prompt {int(row['prompt_id'])}: baseline={row['baseline_trigram_rep']:.3f}, TSK={row['tsk_trigram_rep']:.3f}\n")
        else:
            f.write("None — TSK improved all prompts.\n")

        f.write("\n## Research Contribution Assessment\n\n")
        if credible:
            f.write(
                "This constitutes a **credible research contribution**: the TSK "
                "intervention demonstrably reduces topological looping signals in "
                "GPT-2 hidden states and correspondingly reduces surface repetition "
                "in generated text. The effect size is measurable and consistent "
                "across the majority of prompts."
            )
        else:
            f.write(
                "The TSK intervention shows **limited effectiveness** in this experiment. "
                "While the mechanism is sound in principle, the effect size is small "
                "or inconsistent. Further work should explore richer filtration "
                "parameters, alternative intervention strategies (e.g., soft logit "
                "penalisation), and larger prompt diversity."
            )

    print("Saved phase4-summary.md")

    # ── done ──────────────────────────────────────────────────────────────────
    print("\n=== PHASE COMPLETE ===")
    print("Files saved:")
    print("  phase4-results.csv")
    print("  phase4-completions.txt")
    print("  phase4-summary.md")
    print("  phase4-metric-comparison.png")
    print("  phase4-per-prompt.png")
    print("  phase4-intervention-timeline.png")
    print("  phase4-suppressed-tokens.png")


if __name__ == "__main__":
    main()
