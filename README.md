# Topological-Steering-Kernels

**Topological-Steering-Kernels (TSK)** is a four-phase research pipeline that uses **Topological Data Analysis (TDA)** to detect and suppress looping / repetitive behaviour in autoregressive language model generation.

The core idea is that when a language model is about to produce repetitive output, the geometry of its hidden-state trajectory changes in a measurable way — specifically, the trajectory develops non-trivial *topological holes* (H1 cycles) that persistent homology can quantify. TSK exploits this signal to intervene at decoding time before the model gets stuck in a loop.

---

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [Technical Concepts](#technical-concepts)
3. [Research Hypotheses](#research-hypotheses)
4. [Repository Layout](#repository-layout)
5. [Setup](#setup)
6. [Running the Pipeline](#running-the-pipeline)
7. [Phase Details](#phase-details)
   - [Phase 1 — Data Collection & Loop Detection](#phase-1--data-collection--loop-detection)
   - [Phase 2 — Topological Feature Extraction](#phase-2--topological-feature-extraction)
   - [Phase 3 — Statistical Validation](#phase-3--statistical-validation)
   - [Phase 4 — Topology-Triggered Decoding Intervention](#phase-4--topology-triggered-decoding-intervention)
8. [Figures](#figures)
9. [Pipeline at a Glance](#pipeline-at-a-glance)
10. [Reproducibility](#reproducibility)
11. [Limitations & Future Work](#limitations--future-work)
12. [Notes](#notes)

---

## Background & Motivation

Autoregressive language models such as GPT-2 are well known to degenerate into repetitive loops — endlessly regenerating the same phrase or sentence — especially when seeded with repetitive prompts or when greedy decoding is used without nucleus/top-k sampling.  Existing mitigations (repetition penalty, diverse beam search, top-p sampling) operate purely in the logit space and require careful tuning of heuristic hyperparameters.

**TSK takes a different approach**: rather than modifying the decoding strategy with heuristics, it measures the *shape* of the model's internal computation as generation unfolds.  When the shape of the hidden-state trajectory signals impending repetition (quantified via persistent homology), TSK suppresses the most probable next token and forces the model to choose an alternative.

This repository provides a self-contained experimental pipeline that:

1. Collects GPT-2 hidden states from normal and intentionally looping prompts.
2. Builds a topological feature set using Vietoris-Rips persistent homology.
3. Statistically tests and classifies those features.
4. Deploys a live steering wrapper and measures its effect on generation quality.

---

## Technical Concepts

### Sliding Hidden-State Window

At each generation step the last-layer hidden states of the most recent **10 tokens** are stacked into a 10 × 768 matrix.  This matrix forms a point cloud in 768-dimensional space that is used as the input to the TDA pipeline.

### Vietoris-Rips Filtration

A **Vietoris-Rips complex** is built from the point cloud using the [Gudhi](https://gudhi.inria.fr/) library:

- Each point (hidden state) is a vertex.
- An edge is added between two points when their Euclidean distance is ≤ `max_edge_length` (default **2.0**).
- Higher-dimensional simplices are added similarly up to `max_dimension` (default **2**).
- The filtration sweeps this threshold from 0 to `max_edge_length`, recording when each simplex "appears".

### Persistent Homology

**Persistent homology** tracks the birth and death of topological features as the filtration value grows:

| Dimension | Feature | Intuition |
|-----------|---------|-----------|
| **H0** | Connected components | How isolated the points are |
| **H1** | Loops / 1-cycles | Circular structure in the trajectory |

Each feature is described by a (birth, death) pair.  The **lifetime** `death − birth` measures how persistent (and therefore meaningful) the feature is.

### H1 Features Extracted per Window

| Feature | Description |
|---------|-------------|
| `total_h1` | Sum of all H1 bar lifetimes — total topological "loop energy" |
| `max_h1` | Longest single H1 bar — the most dominant loop |
| `count_h1` | Number of H1 bars with lifetime > 0.01 |
| `entropy_h1` | Shannon entropy of the normalised lifetime distribution |
| `h0_components` | H0 components still alive at filtration value 1.0 (control feature) |

The hypothesis is that **looping windows produce larger H1 features** than normal windows, because repeating the same tokens forces the hidden states to retrace similar directions in activation space, creating closed cycles in the point cloud.

---

## Research Hypotheses

| Hypothesis | Statement |
|-----------|-----------|
| **H1 (TSK model)** | Topological H1 features extracted from hidden-state trajectories are statistically predictive of looping / repetitive generation behaviour. A logistic regression using these features achieves AUC > 0.65 over 5-fold cross-validation, with at least two features significant at p < 0.05 (Mann-Whitney U). |
| **H0 (null / baseline)** | The topological signal is no better than simpler controls (H0-component count alone, or generation step-index), and AUC ≤ 0.55. |

The **verdict** produced by Phase 3 is one of:

- `SUPPORTED` — H1 AUC > 0.65 and ≥ 2 significant features.
- `WEAK` — H1 AUC between 0.55 and 0.65.
- `FALSIFIED` — H1 AUC ≤ 0.55 or no significant features.

---

## Repository Layout

```
Topological-Steering-Kernels/
├── phase1.py          # Data collection, loop scoring, window labelling
├── phase2.py          # Persistent homology feature extraction & plots
├── phase3.py          # Statistical tests, classification, diagnostics
├── phase4.py          # TSK decoding wrapper & evaluation
├── requirements.txt   # Python dependencies
├── figures/           # Pre-generated sample figures (committed)
└── README.md          # This file
```

Runtime-generated artefacts (`.pkl`, `.csv`, `.txt`, `.md` reports, and `.png` figures from the root) are listed in `.gitignore` and are **not committed**.  Only the sample figures in `figures/` are tracked.

---

## Setup

### Prerequisites

- Python 3.10 or later
- A CUDA-capable GPU is optional; the pipeline runs on CPU, but Phase 2 (TDA feature extraction) and Phase 4 (re-running generation) can be 3–5× slower without GPU acceleration.

### Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins the following packages:

| Package | Purpose |
|---------|---------|
| `gudhi` | Vietoris-Rips complex & persistent homology |
| `transformers` | GPT-2 model and tokenizer |
| `torch` | Deep-learning inference |
| `scikit-learn` | Logistic regression, cross-validation, ROC/AUC |
| `matplotlib` | All figure generation |
| `seaborn` | Colour palettes and violin/heatmap plots |
| `pandas` | Feature DataFrames and CSV I/O |
| `numpy` | Array maths |
| `tqdm` | Progress bars |
| `scipy` | Mann-Whitney U test, Shannon entropy |

---

## Running the Pipeline

Phases must be run **in order** from the repository root, because each phase reads artefacts written by the previous one:

```bash
python phase1.py   # ~5–15 min depending on hardware
python phase2.py   # ~5–20 min (TDA is CPU-bound)
python phase3.py   # < 1 min
python phase4.py   # ~10–30 min
```

Each script prints a **GATE PASSED / GATE FAILED** summary to stdout.  If a gate fails, the script explains why and recommends remediation steps before you proceed.

---

## Phase Details

### Phase 1 — Data Collection & Loop Detection

**Script:** `phase1.py`

#### What it does

1. Loads **GPT-2 small** (117 M parameters) from Hugging Face.
2. Generates `GEN_TOKENS = 60` new tokens from each of **20 prompts** (10 factual / "normal" + 10 intentionally repetitive / "looping") using greedy decoding with `output_hidden_states=True`.
3. At every generation step, computes a **trigram loop score**: the fraction of trigrams in the most recent 20-token half that also appear in the preceding 20-token half.
4. Labels each step as `looping = 1` if the score exceeds `LOOP_THRESHOLD = 0.4`, otherwise `normal = 0`.
5. Assembles a **sliding window** of 10 consecutive last-layer hidden states (shape 10 × 768) for every step once enough tokens have been generated.
6. Saves all windows (hidden matrix + label + prompt ID + step index) to `phase1-dataset.pkl`.

#### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WINDOW_SIZE` | 10 | Number of consecutive tokens per window |
| `GEN_TOKENS` | 60 | New tokens generated per prompt |
| `LOOP_THRESHOLD` | 0.4 | Trigram overlap above which a step is labelled `looping` |
| `LOOP_SCORE_WINDOW` | 20 | Size of each half used for trigram comparison |
| `SEED` | 42 | Global random seed for reproducibility |

#### Gate condition

At least **30 %** of all windows must be labelled as looping.  If this threshold is not met the script prints `GATE FAILED` and advises using stronger looping prompts or lowering the overlap threshold.

#### Outputs

| File | Description |
|------|-------------|
| `phase1-dataset.pkl` | Serialised list of window dicts for Phase 2 |
| `phase1-class-balance.png` | Bar chart of normal vs looping window counts |
| `phase1-loop-scores.png` | Trigram loop score over generation steps for 6 representative prompts |
| `phase1-hidden-norms.png` | Heatmap of per-token L2 norms for 5 looping and 5 normal windows |

---

### Phase 2 — Topological Feature Extraction

**Script:** `phase2.py`

#### What it does

1. Loads `phase1-dataset.pkl`.
2. For every window, **normalises** the 10 hidden-state row-vectors to unit length (so distances reflect angular differences rather than magnitude).
3. Constructs a **Vietoris-Rips complex** using Gudhi (`max_edge_length = 2.0`, `max_dimension = 2`).
4. Runs persistent homology and extracts five scalar features per window (see [H1 Features](#h1-features-extracted-per-window)).
5. Saves the full feature table to `phase2-features.csv`.
6. Prints per-class mean/std statistics.
7. Checks the gate: the absolute difference in mean `total_h1` between looping and normal windows must be > 0.05.

#### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_EDGE_LENGTH` | 2.0 | Maximum simplex diameter for Rips filtration |
| `MAX_DIMENSION` | 2 | Maximum homology dimension computed |
| `MIN_LIFETIME` | 0.01 | Bars shorter than this are excluded from `count_h1` |
| `H0_EVAL_VALUE` | 1.0 | Filtration value at which alive H0 components are counted |

#### Outputs

| File | Description |
|------|-------------|
| `phase2-features.csv` | One row per window: `prompt_id`, `step_index`, `loop_label`, and the five TDA features |
| `phase2-persistence-diagrams.png` | Birth/death scatter plots for one representative looping and one normal window |
| `phase2-barcodes.png` | Persistence barcodes (H0 in blue, H1 in red) for the same two windows |
| `phase2-feature-distributions.png` | Violin + strip plots for the four H1 features split by loop label |
| `phase2-correlations.png` | Pearson correlation heatmap of all five features plus `loop_label` |

---

### Phase 3 — Statistical Validation

**Script:** `phase3.py`

#### What it does

1. Loads `phase2-features.csv`.
2. Runs **Mann-Whitney U tests** (two-sided) on each of the four H1 features comparing looping vs normal windows.  Reports U statistic, p-value, and significance.
3. Trains three **logistic regression** classifiers under 5-fold stratified cross-validation:
   - **H1 model** — all five TDA features
   - **H0 baseline** — `h0_components` only
   - **Position baseline** — normalised `step_index` only
4. Reports mean AUC and mean accuracy for each model.
5. Determines the **verdict** (SUPPORTED / WEAK / FALSIFIED) based on H1 AUC and the number of significant Mann-Whitney tests.
6. Writes `phase3-report.md` with full tables and a plain-English conclusion.
7. Produces six diagnostic figures.

#### Verdict logic

```
if H1_AUC > 0.65 AND n_significant_features >= 2:
    verdict = "SUPPORTED"
elif H1_AUC >= 0.55:
    verdict = "WEAK"
else:
    verdict = "FALSIFIED"
```

#### Outputs

| File | Description |
|------|-------------|
| `phase3-report.md` | Written report with Mann-Whitney table, AUC table, and verdict |
| `phase3-roc-curves.png` | Mean ROC curves for all three models over 5-fold CV |
| `phase3-feature-importance.png` | Logistic regression coefficients (positive = associated with looping) |
| `phase3-scatter.png` | `total_h1` vs `max_h1` scatter coloured by label, with 2-feature logistic boundary |
| `phase3-boxplots.png` | Box plots of the four H1 features annotated with p-values |
| `phase3-confusion-matrix.png` | Normalised confusion matrix from the best CV fold |
| `phase3-calibration.png` | Isotonic calibration curve of the H1 logistic model |

---

### Phase 4 — Topology-Triggered Decoding Intervention

**Script:** `phase4.py`

#### What it does

1. Loads the `total_h1` statistics from `phase2-features.csv` and sets the TSK threshold **T** as the **75th percentile** of `total_h1` values among looping windows.
2. Loads GPT-2 small.
3. For each of **15 looping prompts**, generates 60 tokens under two conditions:
   - **Baseline** — standard greedy decoding, no intervention.
   - **TSK-steered** — at each step, computes `total_h1` for the last-10-token window; if `total_h1 > T`, the top-logit token's score is set to `−∞` so the model must choose the next best token instead.
4. Evaluates both completions on three metrics:
   - **Trigram repetition rate** — fraction of trigrams that appear more than once (lower is better under TSK).
   - **Unique-token ratio** — unique tokens / total tokens (higher is better under TSK).
   - **Self-BLEU** — bigram precision of the second half against the first half (lower means less self-repetition; lower is better under TSK).
5. Saves side-by-side completions for five prompts to `phase4-completions.txt`.
6. Writes a plain-English verdict to `phase4-summary.md`.

#### TSK Intervention Mechanism

```
for each generation step:
    hidden_matrix = last_layer_hidden_states[-10:]   # shape (10, 768)
    total_h1 = sum of H1 bar lifetimes (Vietoris-Rips)
    if total_h1 > T:
        logits[argmax(logits)] = -inf   # suppress top token
    next_token = argmax(logits)
```

The intervention is minimal and local: only the single most probable token is suppressed, and only when the topological signal is above the learned threshold.

#### Outputs

| File | Description |
|------|-------------|
| `phase4-results.csv` | Per-prompt metrics for both baseline and TSK conditions |
| `phase4-completions.txt` | Side-by-side text completions for 5 prompts |
| `phase4-summary.md` | Plain-English verdict on whether TSK reduced repetition |
| `phase4-metric-comparison.png` | Grouped bar chart comparing baseline vs TSK across all three metrics |
| `phase4-per-prompt.png` | Dot plot of per-prompt trigram repetition change (baseline → TSK) |
| `phase4-intervention-timeline.png` | `total_h1` trace per generation step for 3 prompts with intervention moments marked |
| `phase4-suppressed-tokens.png` | Bar chart of the 10 most frequently suppressed token strings |

---

## Figures

### Phase 1 — Data Collection & Loop Detection

**Window Class Balance**

Bar chart showing the number of windows labelled as `Normal` (0) vs `Looping` (1).  A healthy class balance with ≥ 30 % looping windows is required before proceeding.

![Phase 1 — Window Class Balance](figures/phase1-class-balance.png)

**Trigram Loop Score over Generation Steps**

The trigram overlap score (0–1) at each generation step for 3 normal prompts (solid blue) and 3 looping prompts (dashed red).  The horizontal dashed red line marks the `LOOP_THRESHOLD = 0.4` above which windows are labelled as looping.  Looping prompts quickly exceed the threshold and stay there.

![Phase 1 — Loop Scores](figures/phase1-loop-scores.png)

**Hidden-State L2 Norm Heatmap**

Side-by-side heatmaps of the L2 norm of the last-layer hidden state at each token position within a window.  Rows are individual windows; columns are the 10 token positions (t-10 … t-1).  Looping windows (left) tend to show more uniform or elevated norm patterns compared to normal windows (right).

![Phase 1 — Hidden Norms](figures/phase1-hidden-norms.png)

---

### Phase 2 — Topological Feature Extraction

**Persistence Diagrams**

Birth/death scatter plots for one looping and one normal window.  Each point is a topological feature; H0 features (connected components) appear in blue, H1 features (1-cycles) in red.  The black dashed diagonal is the line of zero lifetime.  Features far from the diagonal are long-lived and topologically significant.  Looping windows typically show H1 points further from the diagonal.

![Phase 2 — Persistence Diagrams](figures/phase2-persistence-diagrams.png)

**Persistence Barcodes**

The same two windows represented as barcodes.  Each horizontal bar spans from birth to death.  Longer H1 bars (red) in the looping window indicate that closed loops persist over a wider range of the filtration — a direct signature of trajectory self-intersection.

![Phase 2 — Barcodes](figures/phase2-barcodes.png)

**H1 Feature Distributions**

Violin plots (with strip plots overlaid) of the four H1 scalar features split by loop label (0 = Normal, 1 = Looping).  Clear separation between the two distributions supports the TSK hypothesis.

![Phase 2 — Feature Distributions](figures/phase2-feature-distributions.png)

**Feature Correlation Matrix**

Pearson correlation heatmap of all five TDA features plus `loop_label`.  High positive correlations between `total_h1`, `max_h1`, and `loop_label` confirm that looping windows carry more H1 topological structure.

![Phase 2 — Correlations](figures/phase2-correlations.png)

---

### Phase 3 — Statistical Validation

**ROC Curves**

Mean ROC curves over 5-fold stratified cross-validation for the H1 model (all five features, blue), the H0-only baseline (orange dashed), and the position baseline (green dotted).  The shaded area under the H1 model curve visualises its advantage.  AUC > 0.65 for the H1 model is the gate condition to proceed to Phase 4.

![Phase 3 — ROC Curves](figures/phase3-roc-curves.png)

**Feature Importance (Logistic Regression Coefficients)**

Standardised coefficients of the logistic regression trained on all data.  Positive coefficients (green) push predictions toward `looping = 1`; negative (red) push toward `normal = 0`.  `total_h1` and `max_h1` typically dominate with positive coefficients, confirming that high total/max H1 persistence is strongly associated with looping.

![Phase 3 — Feature Importance](figures/phase3-feature-importance.png)

**total\_h1 vs max\_h1 Scatter**

Scatter plot of the two most informative features coloured by true label, with the decision boundary of a 2-feature logistic regression overlaid as a black contour at probability = 0.5.  The separation between classes is visually apparent in this 2D projection.

![Phase 3 — Scatter](figures/phase3-scatter.png)

**Feature Boxplots**

Box plots for each of the four H1 features, with Mann-Whitney U p-values and significance stars (`***` p < 0.001, `**` p < 0.01, `*` p < 0.05, `ns` not significant) annotated above each pair of boxes.

![Phase 3 — Boxplots](figures/phase3-boxplots.png)

**Confusion Matrix**

Normalised confusion matrix from the best-performing cross-validation fold.  Each cell shows the raw count and its row-normalised percentage.  High values on the diagonal indicate the model is correctly distinguishing looping from normal windows.

![Phase 3 — Confusion Matrix](figures/phase3-confusion-matrix.png)

**Calibration Curve**

Calibration curve of the isotonically-calibrated H1 logistic model trained on the full dataset.  The blue step curve should follow the dashed diagonal (perfect calibration), meaning the model's predicted probabilities are reliable estimates of actual looping frequency.

![Phase 3 — Calibration](figures/phase3-calibration.png)

---

### Phase 4 — Topology-Triggered Decoding Intervention

**Metric Comparison (Repetition & Fluency)**

Grouped bar chart comparing the three evaluation metrics (trigram repetition rate, unique-token ratio, self-BLEU) averaged over all 15 looping prompts for baseline GPT-2 (blue) vs TSK-steered GPT-2 (orange).  Successful steering shows lower repetition rate, higher unique-token ratio, and lower self-BLEU under TSK.

![Phase 4 — Metric Comparison](figures/phase4-metric-comparison.png)

**Per-Prompt Repetition: Baseline vs Steered**

Dot plot where each point represents one prompt.  The x-axis is the baseline trigram repetition rate; the y-axis is the TSK repetition rate.  Points below the diagonal indicate prompts where TSK successfully reduced repetition; points above the diagonal indicate the few cases where steering was not helpful.

![Phase 4 — Per-Prompt](figures/phase4-per-prompt.png)

**Intervention Timeline**

`total_h1` traces for 3 representative prompts over 60 generation steps.  Vertical markers show the steps at which TSK intervened (top token suppressed).  The horizontal dashed line shows the threshold T.  TSK fires when the H1 signal spikes, preventing the model from entering or continuing a loop.

![Phase 4 — Intervention Timeline](figures/phase4-intervention-timeline.png)

**Top Suppressed Tokens**

Bar chart of the 10 token strings most frequently suppressed by TSK across all prompts.  These are the tokens the model would repeat but TSK blocked — typically common short words and punctuation that form the backbone of repetitive loops.

![Phase 4 — Suppressed Tokens](figures/phase4-suppressed-tokens.png)

---

## Pipeline at a Glance

```
Prompts (20 normal + 20 looping)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 1 — Data Collection                                        │
│  GPT-2 greedy decoding → hidden states → trigram loop score       │
│  → sliding windows (10 × 768)  →  binary loop labels             │
│  Output: phase1-dataset.pkl                                       │
└───────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 2 — Feature Extraction                                     │
│  Normalise rows → Vietoris-Rips complex → persistent homology     │
│  → {total_h1, max_h1, count_h1, entropy_h1, h0_components}        │
│  Output: phase2-features.csv                                      │
└───────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 3 — Statistical Validation                                 │
│  Mann-Whitney U + logistic regression (5-fold CV)                 │
│  → verdict: SUPPORTED / WEAK / FALSIFIED                          │
│  Output: phase3-report.md + 6 figures                             │
└───────────────────────┬───────────────────────────────────────────┘
                        │  (proceed only if SUPPORTED / WEAK)
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 4 — Decoding Intervention                                  │
│  Threshold T = 75th pct of looping total_h1                       │
│  TSK wrapper: if total_h1 > T → suppress top token               │
│  Evaluate: trigram rep. rate, unique-token ratio, self-BLEU        │
│  Output: phase4-summary.md + 4 figures + completions.txt          │
└───────────────────────────────────────────────────────────────────┘
```

---

## Reproducibility

All phases fix `SEED = 42` for Python's `random`, `numpy`, and `torch` before any computation.  The Gudhi library is deterministic given the same point cloud and filtration parameters.  Scikit-learn classifiers use `random_state=42`.

Running the four scripts on the same hardware with the same Python environment will produce identical numerical results and figures.

> **Note:** GPT-2 model weights are downloaded from Hugging Face on first run and cached locally.  Ensure you have an internet connection, or pre-cache the model beforehand with `huggingface-cli download gpt2` (transformers ≥ 4.0) or by running `from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained("gpt2")` in a Python session.

---

## Limitations & Future Work

| Limitation | Potential Improvement |
|------------|----------------------|
| Only greedy decoding is tested | Evaluate TSK with top-p / top-k sampling |
| Single model (GPT-2 small) | Test on larger models (GPT-2 medium/XL, LLaMA, etc.) |
| Single suppression strategy (top-1 suppression) | Explore soft down-weighting or multi-token suppression |
| Threshold T is set from offline statistics | Learn T adaptively during generation |
| Point cloud only from last layer | Combine hidden states from multiple layers |
| Vietoris-Rips is O(n³) — only feasible for small windows | Explore approximate or witness complexes for larger windows |
| Trigram loop score is a proxy label | Use human annotations or perplexity-based labels |

---

## Notes

- The project is designed as a sequential experiment pipeline; run phases in order so that upstream artefacts are available to downstream phases.
- The figures in the `figures/` directory are representative samples generated from synthetic data.  Run the phase scripts to regenerate them from your own GPT-2 outputs.
- All runtime-generated artefacts (`.pkl`, `.csv`, `.txt`, `.md` reports, root-level `.png` files) are listed in `.gitignore` and will not be committed when you run the scripts.
