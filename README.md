# Federated MIMIC-CXR

A grayscale, multi-label CNN with centralized, FedAvg, and image-level DP experiments.
Requires Python 3.11+.

## Run

    source ~/.virtualenvs/federated-ml/bin/activate
    python -m src.data prepare --mimic-root data/images
    python main.py --mode fedavg --rounds 2
    python main.py --mode all

The lookup CSVs and files/ directory belong under data/images. Preparation filters
to downloaded JPGs before patient sampling and splitting, then caches aspect-ratio-
preserving 128×128 images. Existing cache files are never overwritten. Use a fresh
--cache-dir for another cohort and update cache-dir in pyproject.toml to match.

All modes reuse the same cached patient splits:

| Mode | Training |
| --- | --- |
| centralized | Union of all hospital training sets, without DP |
| fedavg | FedAvg, without DP |
| dp_moderate | FedAvg with target epsilon 8 |
| dp_high | FedAvg with target epsilon 2 |

Use --mode to select one mode or all, and --rounds to override the configured
20 rounds. Centralized training uses the same number of epoch groups and local
epochs as the federated conditions, while preserving its optimizer between groups.

The app configuration keeps cache-dir = "data_cache" and output-dir = "outputs".
The launcher resolves them relative to the project and runs one Flower client at
a time. Training uses MPS on supported Macs, then CUDA or CPU. DP computes
per-image gradients in chunks capped at 8 to bound gradient memory. Noise is drawn
on CPU and transferred to the training device for MPS compatibility.

## Labels and outputs

Five findings follow the order stored in meta.json. Multiple positives are retained;
uncertain labels are masked and unmentioned labels are treated as negative. Images
are normalized to [-1, 1]. Ordinary training uses training-derived class weights;
DP uses fixed unit weights to avoid disclosing private label counts. This weight
difference should be considered when interpreting comparisons.

Each mode saves under outputs/<mode>/:

- best_model.pt: validation-selected weights, configuration, and metadata.
- final_model.pt: weights after the last epoch group/round.
- validation_history.json: validation metrics.
- test_metrics.json: held-out test results for the selected checkpoint.
- privacy.json: per-client spent privacy budgets for DP modes.

Selection uses macro average precision on pooled hospital validation sets. Test
evaluation happens after selection. Repeated runs overwrite the selected mode's
outputs; choose another output-dir to preserve multiple runs.

## DP scope

Native PyTorch computes and clips per-image gradients, adds Gaussian noise, and
performs Poisson sampling. No Opacus dependency is required. Flower continues to
handle all federated rounds and aggregation. Noise is calibrated for the entire
planned run; Flower Context.state preserves each client's step count and noise
multiplier across rounds. Missing or out-of-order
accounting state fails rather than silently resetting the budget. The report uses
the worst client's epsilon, because training partitions are disjoint.

The accountant uses integer-order sampled-Gaussian moments from
[Mironov, Talwar, and Zhang (2019)](https://arxiv.org/abs/1908.10530), section 3.3,
and the standard RDP-to-(epsilon, delta) conversion. Its finite order grid gives
conservative upper bounds. Noise is calibrated separately for each client's
sampling rate and planned optimizer steps. Empty Poisson draws still perform a
noisy update and count toward the budget.

Configure dp-moderate-epsilon, dp-high-epsilon, dp-delta, dp-max-grad-norm, and
dp-physical-batch-size in pyproject.toml. Defaults epsilon 8/2 and delta 1e-5 are
experimental settings, not universal definitions of moderate/strong privacy.

Accounting protects training images, not whole patients. Client partitions and
sizes are treated as fixed/public; held-out validation/test data are outside this
training guarantee. Shared class counts and raw training losses are not released
by DP runs. Sampling and noise use fresh seeds independent of the public experiment
seed. This research implementation uses PyTorch's non-cryptographic RNG.

The prototype notebook used clipping of the averaged batch gradient plus Gaussian
noise. This implementation clips individual image gradients before averaging;
that distinction is needed for its stated image-level privacy accounting.

The bounds describe each DP run separately. Publishing non-DP models alongside
them does not provide a combined DP guarantee. These shared-cache, pooled-validation
experiments simulate hospitals on one machine.

## Checks

    python -m unittest discover -s tests -v

Tests include clipping, analytic accounting cases, cumulative budgets, and all
four training modes using real Flower messages and aggregation in-process.
