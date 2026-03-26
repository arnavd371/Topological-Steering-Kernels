"""
Phase 1 — Data Collection & Loop Detection
===========================================
Loads GPT-2 small, generates text from normal and looping prompts,
captures last-layer hidden states in a sliding window of 10 tokens,
computes a trigram-overlap loop score at each step, labels windows,
and saves everything to phase1-dataset.pkl.

Produces:
  phase1-class-balance.png   — bar chart of looping vs normal window counts
  phase1-loop-scores.png     — loop score over generation steps for 6 prompts
  phase1-hidden-norms.png    — heatmap of L2 norms across token windows
  phase1-dataset.pkl         — list of window dicts for downstream phases
"""

import pickle
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── plot style ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")

# ── constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 10          # sliding window length
GEN_TOKENS = 60           # tokens to generate per prompt
LOOP_THRESHOLD = 0.4      # trigram overlap threshold for loop label
LOOP_SCORE_WINDOW = 20    # tokens in each half for trigram comparison

# ── prompts ───────────────────────────────────────────────────────────────────
NORMAL_PROMPTS = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What causes thunder?",
    "Explain the water cycle in simple terms.",
    "What is the speed of light?",
    "How do airplanes generate lift?",
    "What is the largest planet in our solar system?",
    "How does the human immune system fight viruses?",
    "What is the theory of relativity?",
    "Describe the process of cellular respiration.",
]

LOOPING_PROMPTS = [
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
]


def compute_trigram_overlap(tokens_a: list, tokens_b: list) -> float:
    """Compute fraction of trigrams in tokens_a that also appear in tokens_b."""
    if len(tokens_a) < 3 or len(tokens_b) < 3:
        return 0.0
    trigrams_a = set(zip(tokens_a[:-2], tokens_a[1:-1], tokens_a[2:]))
    trigrams_b = set(zip(tokens_b[:-2], tokens_b[1:-1], tokens_b[2:]))
    if not trigrams_a:
        return 0.0
    return len(trigrams_a & trigrams_b) / len(trigrams_a)


def generate_and_collect(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    prompt_id: int,
    n_tokens: int = GEN_TOKENS,
    window_size: int = WINDOW_SIZE,
) -> tuple[list[dict], list[float]]:
    """
    Auto-regressively generate *n_tokens* new tokens from *prompt*.
    At each step capture the last-layer hidden states for the last
    *window_size* tokens and compute the trigram loop score.

    Returns
    -------
    windows    : list of window dicts (one per step once window is full)
    loop_scores: trigram overlap score at every generation step
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()  # running list of token ids

    # collect hidden states step-by-step
    all_hidden = []    # each element has shape (hidden_dim,); list grows with steps
    loop_scores: list[float] = []
    windows: list[dict] = []

    with torch.no_grad():
        for step in range(n_tokens):
            # forward pass with hidden-state output
            outputs = model(
                torch.tensor([generated], device=device),
                output_hidden_states=True,
            )
            # last-layer hidden state for every token in the current context
            last_hidden = outputs.hidden_states[-1][0]  # (seq_len, 768)

            # record the last token's hidden state
            all_hidden.append(last_hidden[-1].cpu().float().numpy())

            # greedy next token
            next_token = outputs.logits[0, -1].argmax().item()
            generated.append(next_token)

            # ── loop score (trigram overlap between two halves of generated) ─
            gen_new = generated[len(input_ids[0]):]  # only newly generated
            if len(gen_new) >= 2 * LOOP_SCORE_WINDOW:
                half1 = gen_new[-2 * LOOP_SCORE_WINDOW: -LOOP_SCORE_WINDOW]
                half2 = gen_new[-LOOP_SCORE_WINDOW:]
                score = compute_trigram_overlap(half2, half1)
            else:
                score = 0.0
            loop_scores.append(score)
            loop_label = 1 if score > LOOP_THRESHOLD else 0

            # ── sliding window of hidden states ──────────────────────────────
            if len(all_hidden) >= window_size:
                hidden_matrix = np.stack(all_hidden[-window_size:])  # (10, 768)
                windows.append(
                    {
                        "hidden_state_matrix": hidden_matrix,
                        "loop_label": loop_label,
                        "prompt_id": prompt_id,
                        "step_index": step,
                        "token_ids": generated[-window_size:],
                    }
                )

    return windows, loop_scores


def main():
    warnings.filterwarnings("ignore")

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading GPT-2 small …")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    all_windows: list[dict] = []
    # maps prompt_id -> (loop_scores list, condition string)
    prompt_loop_scores: dict[int, tuple[list[float], str]] = {}

    all_prompts = (
        [(p, i, "normal") for i, p in enumerate(NORMAL_PROMPTS)]
        + [(p, 10 + i, "looping") for i, p in enumerate(LOOPING_PROMPTS)]
    )

    # ── generate ──────────────────────────────────────────────────────────────
    for prompt_text, pid, condition in tqdm(all_prompts, desc="Generating"):
        windows, loop_scores = generate_and_collect(
            model, tokenizer, prompt_text, prompt_id=pid
        )
        all_windows.extend(windows)
        prompt_loop_scores[pid] = (loop_scores, condition)

    # ── save dataset ──────────────────────────────────────────────────────────
    with open("phase1-dataset.pkl", "wb") as f:
        pickle.dump(all_windows, f)

    # ── summary statistics ────────────────────────────────────────────────────
    total = len(all_windows)
    n_looping = sum(w["loop_label"] == 1 for w in all_windows)
    n_normal = total - n_looping
    pct_looping = 100.0 * n_looping / total if total > 0 else 0.0

    print("\n=== PHASE 1 SUMMARY ===")
    print(f"Total windows : {total}")
    print(f"Looping       : {n_looping}")
    print(f"Normal        : {n_normal}")
    print(f"% Looping     : {pct_looping:.1f}%")

    # ── gate check ────────────────────────────────────────────────────────────
    if pct_looping >= 30.0:
        print("\nGATE PASSED")
    else:
        print(
            "\nGATE FAILED — DO NOT PROCEED TO PHASE 2\n"
            f"Only {pct_looping:.1f}% of windows are labelled looping "
            f"(need ≥30%). Consider using stronger looping prompts or "
            "lowering the overlap threshold."
        )

    # =========================================================================
    # PLOT 1 — Class balance bar chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Normal", "Looping"],
        [n_normal, n_looping],
        color=sns.color_palette("Set2", 2),
        edgecolor="black",
    )
    for bar, count in zip(bars, [n_normal, n_looping]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total * 0.01, 1),
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Phase 1 — Window Class Balance", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    plt.tight_layout()
    plt.savefig("phase1-class-balance.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 2 — Loop score over time (3 normal + 3 looping)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 5))

    normal_pids = [pid for pid, (_, cond) in prompt_loop_scores.items() if cond == "normal"]
    loop_pids = [pid for pid, (_, cond) in prompt_loop_scores.items() if cond == "looping"]

    rep_normal = sorted(normal_pids)[:3]
    rep_looping = sorted(loop_pids)[:3]

    normal_palette = sns.color_palette("Blues_d", 3)
    looping_palette = sns.color_palette("Reds_d", 3)

    for i, pid in enumerate(rep_normal):
        scores, _ = prompt_loop_scores[pid]
        ax.plot(scores, color=normal_palette[i], label=f"Normal P{pid}", linewidth=1.5)
    for i, pid in enumerate(rep_looping):
        scores, _ = prompt_loop_scores[pid]
        ax.plot(
            scores,
            color=looping_palette[i],
            label=f"Looping P{pid - 10}",
            linewidth=1.5,
            linestyle="--",
        )

    ax.axhline(LOOP_THRESHOLD, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.4)")
    ax.set_title("Phase 1 — Trigram Loop Score over Generation Steps")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Trigram Overlap Score")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("phase1-loop-scores.png", dpi=150)
    plt.close()

    # =========================================================================
    # PLOT 3 — Hidden-state norm heatmap (5 looping + 5 normal)
    # =========================================================================
    rng = np.random.default_rng(SEED)

    looping_windows = [w for w in all_windows if w["loop_label"] == 1]
    normal_windows = [w for w in all_windows if w["loop_label"] == 0]

    n_sample = 5
    sel_loop = rng.choice(len(looping_windows), size=min(n_sample, len(looping_windows)), replace=False)
    sel_norm = rng.choice(len(normal_windows), size=min(n_sample, len(normal_windows)), replace=False)

    # Build norm matrices: each row = one window, each column = one token pos
    def norms(windows_subset, indices):
        mats = []
        for idx in indices:
            h = windows_subset[idx]["hidden_state_matrix"]  # (10, 768)
            norms_row = np.linalg.norm(h, axis=1)           # (10,)
            mats.append(norms_row)
        return np.array(mats)

    loop_norms = norms(looping_windows, sel_loop)
    norm_norms = norms(normal_windows, sel_norm)
    combined = np.vstack([loop_norms, norm_norms])           # (10, 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title in zip(
        axes,
        [loop_norms, norm_norms],
        ["Looping Windows", "Normal Windows"],
    ):
        sns.heatmap(
            data,
            ax=ax,
            cmap="viridis",
            xticklabels=[f"t-{WINDOW_SIZE - i}" for i in range(WINDOW_SIZE)],
            yticklabels=[f"W{j+1}" for j in range(data.shape[0])],
            cbar_kws={"label": "L2 Norm"},
        )
        ax.set_title(f"Phase 1 — Hidden State L2 Norms\n{title}")
        ax.set_xlabel("Token Position in Window")
        ax.set_ylabel("Window Sample")

    plt.tight_layout()
    plt.savefig("phase1-hidden-norms.png", dpi=150)
    plt.close()

    # ── done ──────────────────────────────────────────────────────────────────
    print("\n=== PHASE COMPLETE ===")
    print("Files saved:")
    print("  phase1-dataset.pkl")
    print("  phase1-class-balance.png")
    print("  phase1-loop-scores.png")
    print("  phase1-hidden-norms.png")


if __name__ == "__main__":
    main()
