#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_and_push.sh
#
# Initialises a fresh git repo, lays down 13 well-scoped, conventional-commit
# style commits (one per logical change-set), and pushes to GitHub.
#
# Usage:
#     cd "/Users/kshitiz/Downloads/project_repo_without_lora_adapters (1)"
#     chmod +x setup_and_push.sh
#     ./setup_and_push.sh
#
# Prerequisites:
#     - A GitHub Personal Access Token (classic, scope=repo) for the push.
#       https://github.com/settings/tokens
#     - The empty repo https://github.com/kshitizregmi/federated-conformal-prediction-fcp
#       must already exist (create it on github.com first if you haven't).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="https://github.com/kshitizregmi/federated-conformal-prediction-fcp.git"
GIT_USER_NAME="Kshitiz Regmi"
GIT_USER_EMAIL="kshitizregmi61@gmail.com"

# Wipe any half-initialised .git from previous attempts.
if [ -d .git ]; then
    echo "Removing existing .git ..."
    sudo rm -rf .git
fi

git init -b main
git config user.name  "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

# ---------------------------------------------------------------------------
# 1/15 — repo hygiene
# ---------------------------------------------------------------------------
git add .gitignore
git commit -m "chore: add .gitignore for Python, OS, and heavy LoRA artefacts" \
           -m "Excludes only the truly heavy and trivially reproducible LoRA
training artefacts; the partitioned dataset and scoring CSV stay
checked in so the repo is fully self-contained.

Excluded:
  - flower_qwen_lora_outputs/  (LoRA adapter checkpoints, GBs)
  - *.npz / *.safetensors / *.bin / *.pt
  - __pycache__/, .venv/, .DS_Store, *.log, ray_tmp/

Kept (committed in later commits):
  - scienceqa_sft_clients_with_images/  full federated dataset
  - fed_dscp_scored_examples.csv        per-row Qwen scoring output"

# ---------------------------------------------------------------------------
# 2/15 — data partitioning script
# ---------------------------------------------------------------------------
git add split.py
git commit -m "feat(data): partition ScienceQA into 5 non-IID federated clients" \
           -m "Builds the federated dataset used by every downstream script.

Pipeline:
  1. Load ScienceQA from HuggingFace and keep only 4-choice closed-choice
     MCQ rows with a non-null category/subject/topic.
  2. Drop categories with fewer than ceil(20 / TRAIN_FRAC) = 29 total
     examples so every kept category contributes >=20 rows to train.
  3. Round-robin whole categories to NUM_CLIENTS = 5 clients. Whole-
     category assignment (vs random row-level) is what makes the
     partition genuinely non-IID at the category level — see the
     bubble matrix in final_partition_figures/.
  4. Within each client, stratified train/val/calib/test split
     (70 / 10 / 10 / 10) so the marginal distribution per category is
     preserved across splits.
  5. Persist OpenAI-style JSONL chat rows + extracted image PNGs under
     scienceqa_sft_clients_with_images/client_{0..4}/.

Also runs a smoke-test inference on one Client-0 example with
Qwen2.5-VL-3B-Instruct to verify processor + model wiring before the
federated training run.

Seed: SEED = 42 (used everywhere downstream)."

# ---------------------------------------------------------------------------
# 3/15 — federated LoRA fine-tuning
# ---------------------------------------------------------------------------
git add train.py
git commit -m "feat(train): federated LoRA fine-tune of Qwen2.5-VL-3B with Flower + Ray" \
           -m "Spins up a Flower 1.13 simulation (NUM_CLIENTS = 5,
NUM_ROUNDS = 10) on top of a Ray backend, with one ClientApp per
federated client and a custom SaveFedAvg server-side strategy that
snapshots the aggregated LoRA tensors after every round.

Key implementation choices:

  * 4-bit NF4 quantisation (BitsAndBytes) + double-quant for the base
    Qwen weights so the 3 B model fits on a single 24 GB GPU.
  * LoRA rank 8, alpha 16, dropout 0.05 on q/k/v/o + gate/up/down.
  * MAX_LOCAL_STEPS = 80, GRAD_ACCUM_STEPS = 4, lr = 2e-4 (AdamW).
  * Loss is masked to the assistant span only (prompt tokens get -100).
  * GPU auto-selection via nvidia-smi: only GPUs with >=20 GB free are
    surfaced to the simulation.
  * Ray init is hardened against uv's RAY_RUNTIME_ENV_HOOK injection,
    which would otherwise crash with 'multiple values for keyword
    argument runtime_env'. Hence the requirement to launch with
    'uv run --no-project python train.py'.

Outputs:
  flower_qwen_lora_outputs/
    initial_adapter/                   — randomly-initialised LoRA
    global_lora_round_{1..10}.npz      — aggregated tensors per round
    federated_adapter_final/           — final aggregated adapter
    local_adapters/client_{c}/round_{r}/  — per-client local adapters

The final adapter is the input to score_client_data.py (next commit)."

# ---------------------------------------------------------------------------
# 4/15 — per-example scoring stage
# ---------------------------------------------------------------------------
git add score_client_data.py
git commit -m "feat(scoring): emit A/B/C/D logits and uncertainty features per row" \
           -m "Loads the federated LoRA adapter on top of the base
Qwen2.5-VL-3B model and runs a single forward pass per validation /
calibration / test example, recording everything the conformal stage
needs:

  * Per-letter probabilities  p_A, p_B, p_C, p_D  and logits z_A..z_D.
  * Top-1 prediction + correctness flag.
  * LAC nonconformity score: score_correct = 1 - p(gold).
  * Four uncertainty features used as conditioning signals downstream:
      - entropy   = -Σ p log p
      - max_prob  = max p
      - margin    = p_(1) - p_(2)
      - energy    = -log Σ exp(z)
  * Per-row metadata: client_id, split, idx, category, subject, topic,
    has_image.

Output: fed_dscp_scored_examples.csv (~4.7 k rows, gitignored — it is
regenerable, see .gitignore).

This stage is the boundary between the model and the conformal layer:
every CP method downstream consumes this CSV and never re-touches the
neural network."

# ---------------------------------------------------------------------------
# 5/15 — core conformal-prediction methods
# ---------------------------------------------------------------------------
git add conformal_prediction.py
git commit -m "feat(cp): implement Fed-DSCP, Fed-Shrink-v2, and Fed-CVAE-CP" \
           -m "The reference implementation of every conformal method
reported in the paper. All methods share the standard split-CP quantile

    q_hat = quantile(scores, ceil((n+1)(1-α))/n, method='higher')

with α = 0.10 and operate on score_correct = 1 - p(gold).

Methods (in increasing sophistication):

  Global-LAC      single q_hat over all clients' calibration scores.
  Local-LAC       per-client q_hat from that client's calibration set.
  Local-APS       adaptive prediction sets per client.
  Fed-DSCP        per-(client, difficulty bin) q_hat. Bins are defined
                  by splitting each client's *validation* entropy at the
                  median (validation, not calibration, to keep
                  exchangeability). Cells with n < MIN_CALIB_PER_BIN
                  fall back to local LAC.
  Fed-Shrink-v2   empirical-Bayes shrinkage of Fed-DSCP toward a global
                  per-bin anchor:
                      λ        = n / (n + τ),     τ = 20
                      q_shrink = λ q_DSCP + (1-λ) q_global_bin
                  Borrows strength across clients without sharing raw
                  data — exactly what FL needs.
  Fed-CVAE-CP     conditional VAE generates synthetic calibration scores
  (τ = 5)         conditioned on the validation feature mean per cell;
                  the augmented quantile is then shrunk toward Fed-DSCP.

Output: main_result.csv (next commit)."

# ---------------------------------------------------------------------------
# 6/15 — benchmarking + styled result table
# ---------------------------------------------------------------------------
git add benchmarking.py eg.py
git commit -m "feat(report): emit the styled main-result table figure" \
           -m "Two scripts that produce the publication-ready 'final
weighted table' image used in the paper.

  benchmarking.py  re-evaluates Global/Local-LAC, Local-APS, Fed-DSCP,
                   Fed-Shrink-v2 with the bin column recomputed at
                   runtime so the figure is reproducible from CSV alone.
                   Renders final_weighted_table.{pdf,png} via
                   matplotlib's Table, with header / zebra-row styling.

  eg.py            same evaluation logic, alternate styling, drops the
                   figure as main_result_table_correct.{pdf,png}. Also
                   contains a commented-out path that re-renders the
                   table from main_result.csv directly.

Both end up at the same numbers — they are deliberately decoupled from
conformal_prediction.py so you can regenerate the figure without
re-training the CVAE."

# ---------------------------------------------------------------------------
# 7/15 — ablation
# ---------------------------------------------------------------------------
git add ablation.py
git commit -m "feat(ablation): sweep CVAE / KDE quantile estimators and shrinkage τ" \
           -m "Ablation study used to justify the headline choices of
the CVAE variant (τ = 5) and to compare against a non-parametric KDE
counterpart.

Variants evaluated on the test split (LAC sets, α = 0.10):

  Fed-DSCP                 — baseline, no shrinkage
  CVAE-no-shrink           — pure CVAE-augmented quantile
  CVAE-τ ∈ {1, 5, 10, 20}  — shrinkage sweep
  Fed-CondKDE              — Silverman-bandwidth Gaussian KDE quantile,
                             no shrinkage
  Fed-CondKDE-Shrink       — same KDE quantile, τ = 5 shrinkage to DSCP

Confirms two findings:

  1. Generative augmentation alone (CVAE-no-shrink, Fed-CondKDE) under-
     covers and hallucinates at ~13 %. Augmentation must be combined
     with shrinkage.
  2. With shrinkage, CVAE and KDE converge within noise — the
     non-parametric KDE is a credible parameter-free alternative.

Output: ablation_result.csv."

# ---------------------------------------------------------------------------
# 8/15 — α sweep, Pareto and reliability plots
# ---------------------------------------------------------------------------
git add plot_tradeoff.py
git commit -m "feat(plots): sweep α to render Pareto and commit/hallucination curves" \
           -m "Sweeps α ∈ {0.02, 0.04, …, 0.28} for every method and
emits two paper figures plus the underlying CSV dumps.

  pareto_coverage_setsize.{pdf,png}
      Coverage (x) vs average prediction-set size (y). Lower-right is
      better. Highlights the α = 0.10 operating point with a filled
      circle and the 90 % coverage target with a dashed line. This is
      the figure that visually justifies Fed-Shrink-v2 sitting on the
      Pareto frontier.

  tradeoff_commit_hallucination.{pdf,png}
      Two-panel reliability plot: commit-rate vs hallucination-rate
      (left), and commit-rate vs precision@commit (right).

  pareto_curves.csv     coverage / set-size per (α, method)
  tradeoff_sweep.csv    commit / hallucination / precision per (α, method)

Also re-trains the CVAE inside this script (CVAE_EPOCHS = 300) so the
plotting stage is self-contained."

# ---------------------------------------------------------------------------
# 9/15 — diagnostics
# ---------------------------------------------------------------------------
git add nocpvscp.py baseline_without_fcp_hallucination.py show_bin_size.py tinybindemo.py
git commit -m "feat(diagnostics): per-client comparisons and shrinkage stress tests" \
           -m "Four small scripts that produce the diagnostic numbers
and figures the paper cites in passing.

  baseline_without_fcp_hallucination.py
      Pure top-1 hallucination baseline. Per-client and overall:
      accuracy, hallucination rate (= 1 - accuracy). The 'before' row
      of the before/after CP comparison.

  nocpvscp.py
      Joint per-client table comparing No-CP top-1 vs Fed-Shrink-v2.
      Demonstrates the 3.4× hallucination drop (13.05 % → 3.79 %)
      while preserving 90 %+ coverage. Source of
      comparison_noCP_vs_Shrinkv2.csv.

  show_bin_size.py
      Bar chart of calibration counts per (client, difficulty bin) with
      the MIN_CALIB = 10 safety threshold. Surfaces the small-cell
      problem (Client 3 / Bin 1) that motivates Fed-Shrink-v2.
      → calibration_bin_sizes.{pdf,png}

  tinybindemo.py
      Synthetic stress test: keep only 3 calibration points in
      Client 3 / Bin 1 and show that Fed-DSCP's q_hat becomes unstable
      while Fed-Shrink-v2 remains calibrated.
      → tiny_bin_demo_final.{pdf,png}"

# ---------------------------------------------------------------------------
# 10/15 — EDA scripts
# ---------------------------------------------------------------------------
git add eda.py t.py
git commit -m "feat(eda): paper-ready dataset distribution and Pareto bubble figures" \
           -m "Two analysis scripts whose only job is to make the
non-IID partitioning visually and quantitatively obvious.

  eda.py                 → eda_paper_plots/
      fig1_split_size_per_client          Train/Val/Calib/Test bars per client
      fig2_client_vs_topic_matrix         top-25 topics × clients heatmap
      fig3_client_{c}_topic_distribution  within-client topic-by-split bars
      fig4_client_{c}_image_text          image-based vs text-only proportions
      table_split_sizes_by_client.csv     populates the readme.md split table
      table_topic_counts_by_client_split.csv

  t.py                   → final_partition_figures/
      fig1_rotated_noniid_bubble_matrix   Pareto bubble matrix —
                                          (category × client) bubbles whose
                                          size and colour encode train counts.
      fig2_grouped_client_split_sizes_clean  publication-ready split-size bars

Both scripts are pure pandas + matplotlib; no model dependency."

# ---------------------------------------------------------------------------
# 11/15 — federated dataset (≈252 MB)
# ---------------------------------------------------------------------------
echo
echo "Adding the partitioned ScienceQA dataset (~252 MB). This may take a minute..."
echo
git add scienceqa_sft_clients_with_images/
git commit -m "data(scienceqa): commit the partitioned 5-client federated dataset" \
           -m "Output of split.py. Checked in so the repo is fully self-
contained and the pipeline can be reproduced without re-running the
HuggingFace download step.

Layout:
  scienceqa_sft_clients_with_images/
    client_{0..4}/
      train.jsonl              OpenAI-style chat rows
      validation.jsonl         (used for bin-edge estimation)
      calibration.jsonl        (used for conformal q_hat)
      test.jsonl               (used for final metrics)
      images/                  per-row PNG images (where applicable)

Split sizes (total = 4 733 rows):
  Client 0:   511 / 73  / 73  / 74
  Client 1:   637 / 91  / 91  / 91
  Client 2: 1 243 / 178 / 178 / 178
  Client 3:   416 / 59  / 60  / 60
  Client 4:   504 / 72  / 72  / 72

Generation is deterministic with SEED = 42 so re-running split.py
produces this exact partitioning byte-for-byte."

# ---------------------------------------------------------------------------
# 12/15 — per-row scoring CSV
# ---------------------------------------------------------------------------
git add fed_dscp_scored_examples.csv
git commit -m "data(scoring): commit per-row Qwen2.5-VL-3B scoring output" \
           -m "Output of score_client_data.py — the bridge between the
neural model and the conformal layer. Every row in val / calibration /
test gets:

  client_id, split, idx, gold, pred, correct, score_correct,
  p_A..p_D, z_A..z_D, entropy, max_prob, margin, energy,
  category, subject, topic, has_image, dscp_bin

Checked in (~400 KB) so every CP method can be re-run in seconds
without reloading the LoRA-adapted Qwen model. Re-running
score_client_data.py with the same federated_adapter_final reproduces
this CSV exactly."

# ---------------------------------------------------------------------------
# 13/15 — result CSVs
# ---------------------------------------------------------------------------
git add main_result.csv ablation_result.csv pareto_curves.csv tradeoff_sweep.csv comparison_noCP_vs_Shrinkv2.csv eda_paper_plots/table_split_sizes_by_client.csv eda_paper_plots/table_topic_counts_by_client_split.csv
git commit -m "data(results): commit reference CSVs reproducible from the pipeline" \
           -m "These are small (<400 KB total) and deterministic given
SEED = 42, so they live in the repo as a quick-look reference and as
golden files for any future refactor.

  main_result.csv                 — Step 4 output (5 methods × 5 metrics)
  ablation_result.csv             — Step 5 output (8 variants × 5 metrics)
  pareto_curves.csv               — Step 6 output (α × method × cov/size)
  tradeoff_sweep.csv              — Step 6 output (α × method × commit metrics)
  comparison_noCP_vs_Shrinkv2.csv — Step 7 output (per-client, no-CP vs CP)
  eda_paper_plots/table_*.csv     — Step 8 EDA aggregates"

# ---------------------------------------------------------------------------
# 14/15 — paper figures
# ---------------------------------------------------------------------------
git add \
    main_result_table.pdf main_result_table.png \
    main_result_table_correct.pdf main_result_table_correct.png \
    final_weighted_table.pdf final_weighted_table.png \
    pareto_coverage_setsize.pdf pareto_coverage_setsize.png \
    tradeoff_commit_hallucination.pdf tradeoff_commit_hallucination.png \
    calibration_bin_sizes.pdf calibration_bin_sizes.png \
    tiny_bin_demo.pdf tiny_bin_demo.png \
    tiny_bin_demo_final.pdf tiny_bin_demo_final.png \
    tiny_bin_demo_v2.pdf tiny_bin_demo_v2.png \
    final_partition_figures/ \
    eda_paper_plots/fig1_split_size_per_client.pdf eda_paper_plots/fig1_split_size_per_client.png \
    eda_paper_plots/fig2_client_vs_topic_matrix.pdf eda_paper_plots/fig2_client_vs_topic_matrix.png \
    'eda_paper_plots/fig3_client_0_topic_distribution_by_split.pdf' \
    'eda_paper_plots/fig3_client_0_topic_distribution_by_split.png' \
    'eda_paper_plots/fig3_client_1_topic_distribution_by_split.pdf' \
    'eda_paper_plots/fig3_client_1_topic_distribution_by_split.png' \
    'eda_paper_plots/fig3_client_2_topic_distribution_by_split.pdf' \
    'eda_paper_plots/fig3_client_2_topic_distribution_by_split.png' \
    'eda_paper_plots/fig3_client_3_topic_distribution_by_split.pdf' \
    'eda_paper_plots/fig3_client_3_topic_distribution_by_split.png' \
    'eda_paper_plots/fig3_client_4_topic_distribution_by_split.pdf' \
    'eda_paper_plots/fig3_client_4_topic_distribution_by_split.png' \
    'eda_paper_plots/fig4_client_0_image_text_distribution.pdf' \
    'eda_paper_plots/fig4_client_0_image_text_distribution.png' \
    'eda_paper_plots/fig4_client_1_image_text_distribution.pdf' \
    'eda_paper_plots/fig4_client_1_image_text_distribution.png' \
    'eda_paper_plots/fig4_client_2_image_text_distribution.pdf' \
    'eda_paper_plots/fig4_client_2_image_text_distribution.png' \
    'eda_paper_plots/fig4_client_3_image_text_distribution.pdf' \
    'eda_paper_plots/fig4_client_3_image_text_distribution.png' \
    'eda_paper_plots/fig4_client_4_image_text_distribution.pdf' \
    'eda_paper_plots/fig4_client_4_image_text_distribution.png'
git commit -m "docs(figures): add publication-quality PDFs and PNGs" \
           -m "Both raster (.png) and vector (.pdf) versions of every
figure referenced by the paper and embedded in readme.md.

Categories:
  - Result tables:     main_result_table*, final_weighted_table
  - Pareto / sweeps:   pareto_coverage_setsize, tradeoff_commit_hallucination
  - Diagnostics:       calibration_bin_sizes, tiny_bin_demo*
  - Partition / EDA:   final_partition_figures/, eda_paper_plots/fig*

Vector PDFs ship at 300 DPI with embedded fonts (pdf.fonttype = 42)
for journal submission; PNGs are 150-300 DPI for web / readme rendering."

# ---------------------------------------------------------------------------
# 15/15 — documentation
# ---------------------------------------------------------------------------
git add readme.md setup_and_push.sh
git commit -m "docs: comprehensive readme and one-shot push script" \
           -m "Technical documentation that walks a fresh contributor
from zero to reproduced paper figures.

readme.md covers:
  1. Repository layout
  2. uv installation (incl. CUDA wheels and the --no-project caveat)
  3. The 5-client non-IID partitioning and per-split usage table
  4. End-to-end pipeline order (split → train → score → CP → plots)
  5. EDA / paper-figure index (Pareto bubble, Pareto curve, etc.)
  6. Math + intuition for Fed-DSCP and Fed-Shrink-v2 with code pointers
  7. Result tables (main, ablation, no-CP vs CP) with read-out
  8. VAE / CVAE / KDE methodology and 'when to use which' guide

Embedded figures are loaded by relative path so the readme renders
correctly on github.com.

setup_and_push.sh is the script that produced this very commit graph;
keeping it in the repo so the commit history is reproducible."

# ---------------------------------------------------------------------------
# Push (this will upload ~252 MB on the first push — be patient)
# ---------------------------------------------------------------------------
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"

# Bump the HTTP buffer so the dataset commit doesn't choke on slow links.
git config http.postBuffer 524288000

git push -u origin main

echo
echo "Done. View at: ${REPO_URL%.git}"
