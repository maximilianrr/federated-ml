# Federated MIMIC-CXR

A small grayscale CNN trained with Flower FedAvg on patient-disjoint simulated
hospitals. Requires Python 3.11+.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m src.data prepare --mimic-root /path/to/mimic-cxr-jpg
python main.py --rounds 2
python main.py
```

Preparation caches 128×128 images with aspect-ratio-preserving padding. Use
`--max-patients 2000` for a pilot, `--image-size 224` for a resolution experiment,
and a fresh `--cache-dir` for each prepared cohort. Preparation refuses to overwrite
existing cache files; fix unreadable images and use a fresh directory after a failure.
Inspect the cache with `python -m src.data info`.

Configuration lives in `pyproject.toml`: `cache-dir = "data_cache"` and
`output-dir = "outputs"`. The launcher resolves these relative to the project and
schedules one client at a time. PyTorch uses MPS on supported Macs, then CUDA or CPU.
Defaults are batch size 32, one local epoch, learning rate 0.001, and 20 rounds.

## Labels and evaluation

The five outputs follow the class order stored in `meta.json`. Multiple findings
are retained; uncertain labels (`-1`) are masked, and unmentioned labels are treated
as negative. This is an explicit modeling assumption. Training-only class counts
supply bounded positive weights; image normalization is fixed to `[-1, 1]`.

Validation uses the pooled hospital validation sets. The best checkpoint is selected
by macro average precision and evaluated on the global test set only after training.
Undefined metrics for findings without both positive and negative examples are
omitted; validation requires both values for every finding.

`outputs/` contains `best_model.pt` (weights plus metadata), `final_model.pt`
(weights), `validation_history.json`, and `test_metrics.json`. Repeated runs replace
these files; change `output-dir` to keep separate experiments.

The shared cache and pooled validation are for a local simulation. Keep the same
prepared cohort for centralized/FedAvg comparisons. The training code does not
implement differential privacy.

## Checks

```bash
python -m unittest discover -s tests -v
```
