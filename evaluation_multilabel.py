"""
Multilabel evaluation utilities for the five-label medical-imaging experiment.

This script has two uses:
1. Summarize the test_metrics.json and validation_history.json files already
   present in outputs/{centralized,fedavg,dp_moderate,dp_high}.
2. Re-evaluate a PyTorch model on a DataLoader using multilabel metrics.

For multilabel classification, targets/logits have shape [n_samples, n_labels].
The model should return raw logits; sigmoid is applied independently to each label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
        f1_score,
        hamming_loss,
    )
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required. Install it with: pip install scikit-learn"
    ) from exc


def _safe_float(value):
    """Convert NumPy scalar values to JSON-safe Python floats."""
    return float(value) if value is not None and np.isfinite(value) else None


def multilabel_metrics(
    y_true,
    probabilities,
    threshold: float = 0.5,
    label_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Compute per-label and aggregate metrics for a multilabel task."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 2 or probabilities.ndim != 2:
        raise ValueError(
            "Expected y_true and probabilities to have shape "
            "(n_samples, n_labels); do not flatten multilabel targets."
        )
    if y_true.shape != probabilities.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"probabilities={probabilities.shape}"
        )
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Targets must be binary 0/1 values for each label.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be in [0, 1]; apply sigmoid to logits first.")

    y_true = y_true.astype(int)
    y_pred = (probabilities >= threshold).astype(int)
    n_labels = y_true.shape[1]
    names = list(label_names) if label_names is not None else [
        f"label_{i}" for i in range(n_labels)
    ]
    if len(names) != n_labels:
        raise ValueError(f"Got {len(names)} label names for {n_labels} labels.")

    per_label = {}
    aurocs, aps = [], []
    for j, name in enumerate(names):
        truth = y_true[:, j]
        prob = probabilities[:, j]
        pred = y_pred[:, j]
        positives = int(truth.sum())
        count = int(len(truth))

        # AUROC is undefined if this evaluation subset has only positives or only negatives.
        auc = _safe_float(roc_auc_score(truth, prob)) if np.unique(truth).size == 2 else None
        ap = _safe_float(average_precision_score(truth, prob)) if positives > 0 else None
        if auc is not None:
            aurocs.append(auc)
        if ap is not None:
            aps.append(ap)

        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, pred, average="binary", zero_division=0
        )
        per_label[name] = {
            "count": count,
            "positives": positives,
            "prevalence": _safe_float(truth.mean()),
            "precision_at_threshold": _safe_float(precision),
            "recall_at_threshold": _safe_float(recall),
            "f1_at_threshold": _safe_float(f1),
            "auroc": auc,
            "average_precision": ap,
            "threshold": float(threshold),
        }

    # Brier score averages squared probability error over all sample-label pairs.
    brier = float(np.mean((probabilities - y_true) ** 2))
    eps = 1e-7
    clipped = np.clip(probabilities, eps, 1 - eps)
    # Binary cross-entropy / negative log likelihood, averaged over sample-label pairs.
    nll = float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))

    return {
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(n_labels),
        "threshold": float(threshold),
        "macro_auroc": _safe_float(np.mean(aurocs)) if aurocs else None,
        "macro_average_precision": _safe_float(np.mean(aps)) if aps else None,
        "micro_f1_at_threshold": _safe_float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_f1_at_threshold": _safe_float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "hamming_loss_at_threshold": _safe_float(hamming_loss(y_true, y_pred)),
        "brier_score": brier,
        "binary_cross_entropy": nll,
        "per_label": per_label,
    }


def evaluate_multilabel_model(
    model,
    loader,
    device,
    criterion=None,
    label_names: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate a model on a DataLoader yielding (images, multi-hot targets).

    Assumes model(images) returns raw logits with shape [batch, n_labels],
    and targets have the same shape with binary 0/1 values.
    """
    import torch

    was_training = model.training
    model.eval()
    model.to(device)
    all_targets, all_probs, losses = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).float()
            logits = model(images)
            if logits.ndim != 2 or targets.ndim != 2:
                raise ValueError(
                    "Expected model outputs and targets shaped [batch, n_labels]. "
                    "This evaluator is for multilabel classification."
                )
            if logits.shape != targets.shape:
                raise ValueError(
                    f"Logit/target shape mismatch: {tuple(logits.shape)} vs "
                    f"{tuple(targets.shape)}"
                )
            probs = torch.sigmoid(logits)
            all_targets.append(targets.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            if criterion is not None:
                losses.append(float(criterion(logits, targets).item()))

    model.train(was_training)
    if not all_targets:
        raise ValueError("The DataLoader yielded no batches.")

    report = multilabel_metrics(
        np.concatenate(all_targets),
        np.concatenate(all_probs),
        threshold=threshold,
        label_names=label_names,
    )
    if losses:
        report["mean_batch_loss"] = float(np.mean(losses))
    return report


def load_existing_outputs(outputs_dir: Path) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Load the JSON metrics and validation histories already saved by training."""
    conditions = {}
    histories = {}
    for condition_dir in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        metrics_path = condition_dir / "test_metrics.json"
        history_path = condition_dir / "validation_history.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            conditions[condition_dir.name] = metrics
        if history_path.exists():
            raw = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                histories[condition_dir.name] = raw
            elif isinstance(raw, dict):
                # Accept a wrapped history format too.
                histories[condition_dir.name] = raw.get("history", raw.get("metrics", []))

    if not conditions:
        raise FileNotFoundError(
            f"No */test_metrics.json files found under {outputs_dir.resolve()}"
        )

    rows = []
    for name, m in conditions.items():
        row = {
            "condition": name,
            "loss": m.get("loss"),
            "evaluated_labels": m.get("evaluated_labels"),
            "macro_auroc": m.get("macro_auroc"),
            "macro_average_precision": m.get("macro_ap", m.get("macro_average_precision")),
        }
        label_ids = sorted(
            {
                key.rsplit("_", 1)[0]
                for key in m
                if key.startswith("label_") and key.endswith("_auroc")
            },
            key=lambda x: int(x.split("_")[1]),
        )
        for label_id in label_ids:
            row[f"{label_id}_auroc"] = m.get(f"{label_id}_auroc")
            row[f"{label_id}_ap"] = m.get(f"{label_id}_ap")
        rows.append(row)
    return pd.DataFrame(rows), histories


def plot_saved_test_metrics(table: pd.DataFrame, out_dir: Path) -> None:
    """Plot metrics from saved test_metrics.json files (no model/data loading needed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = table["condition"].tolist()

    for metric, ylabel, filename, title in [
        ("macro_auroc", "Macro AUROC", "macro_auroc_comparison.png", "Test macro AUROC"),
        ("macro_average_precision", "Macro average precision", "macro_ap_comparison.png",
         "Test macro average precision"),
        ("loss", "Saved test loss", "test_loss_comparison.png", "Saved test loss"),
    ]:
        if metric not in table or table[metric].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        values = table[metric].astype(float)
        ax.bar(names, values)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    auc_cols = [c for c in table.columns if c.startswith("label_") and c.endswith("_auroc")]
    ap_cols = [c for c in table.columns if c.startswith("label_") and c.endswith("_ap")]
    for cols, metric_name, filename in [
        (auc_cols, "AUROC", "per_label_auroc.png"),
        (ap_cols, "Average precision", "per_label_average_precision.png"),
    ]:
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(cols))
        width = 0.8 / max(len(names), 1)
        for i, (_, row) in enumerate(table.iterrows()):
            vals = [row[c] for c in cols]
            ax.bar(x + i * width, vals, width=width, label=row["condition"])
        ax.set_xticks(x + width * (len(names) - 1) / 2)
        ax.set_xticklabels([c.rsplit("_", 1)[0] for c in cols])
        ax.set_ylabel(metric_name)
        ax.set_title(f"Per-label test {metric_name}")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_validation_history(histories: Mapping[str, list[dict]], out_dir: Path) -> None:
    """Plot validation loss, macro AUROC and macro AP over saved rounds/epochs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("loss", "Validation loss", "validation_loss.png"),
        ("macro_auroc", "Validation macro AUROC", "validation_macro_auroc.png"),
        ("macro_ap", "Validation macro average precision", "validation_macro_ap.png"),
    ]
    for key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for name, records in histories.items():
            xs, ys = [], []
            for i, record in enumerate(records):
                value = record.get(key)
                if value is None and key == "macro_ap":
                    value = record.get("macro_average_precision")
                if value is not None:
                    xs.append(record.get("round", record.get("epoch", i)))
                    ys.append(value)
            if ys:
                ax.plot(xs, ys, marker="o", markersize=3, label=name)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Round / epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by condition")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def summarize_saved_outputs(outputs_dir: str | Path = "outputs",
                            report_dir: str | Path = "evaluation_results") -> pd.DataFrame:
    """Create a CSV/Markdown/JSON summary and plots from the existing output JSONs."""
    outputs_dir, report_dir = Path(outputs_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    table, histories = load_existing_outputs(outputs_dir)
    table.to_csv(report_dir / "results_table.csv", index=False)
    table.round(5).to_markdown(index=False)
    (report_dir / "results_table.md").write_text(
        table.round(5).to_markdown(index=False), encoding="utf-8"
    )
    (report_dir / "results_table.json").write_text(
        json.dumps(table.replace({np.nan: None}).to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    plot_saved_test_metrics(table, report_dir)
    plot_validation_history(histories, report_dir)
    print(f"Saved evaluation summary to: {report_dir.resolve()}")
    print(table.round(4).to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate/summarize saved outputs for a five-label multilabel task."
    )
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Directory containing centralized/, fedavg/, dp_moderate/, dp_high/.")
    parser.add_argument("--report-dir", default="evaluation_results",
                        help="Directory where CSV/Markdown/plots will be saved.")
    args = parser.parse_args()
    summarize_saved_outputs(args.outputs_dir, args.report_dir)


if __name__ == "__main__":
    main()
