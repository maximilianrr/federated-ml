"""
Evaluation metrics and plots for the federated / DP / membership-inference study.

    utility  : accuracy, macro-F1, MCC, kappa, macro one-vs-rest AUROC, per-class P/R/F1,
               confusion matrix, calibration (ECE, Brier, NLL), generalization gap,
               cross-client fairness (worst client, spread)

    privacy  : AUROC (+ bootstrap CI), TPR @ low FPR, calibrated attack accuracy (attack_sim),
               MIA advantage, empirical epsilon lower bound, paired significance tests,
               utility retention vs leakage reduction relative to a baseline
"""

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg") #figures saved to disk, can be changed of course if you want to view the plots
import matplotlib.pyplot as plt

from scipy.stats import beta as _beta

from attack_sim import auroc, roc_curve

# ----------------------------------------------------------------------------------------
# utility metrics
# ----------------------------------------------------------------------------------------


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Rows are the true class, columns the predicted class."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_report(y_true, y_pred, num_classes: int,
                          class_names: Optional[Sequence[str]] = None) -> dict:
    """
    Accuracy, macro / weighted F1, balanced accuracy and per-class precision, recall, F1.

    Macro-F1 gives every class the same weight, which matters for imbalanced medical labels
    where plain accuracy can look good while rare findings are missed.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    total = float(cm.sum())
    pe = float((support * predicted).sum() / max(total ** 2, 1))
    kappa = (tp.sum() / max(total, 1) - pe) / (1 - pe) if pe < 1 else 0.0
    mcc_den = np.sqrt(max(total ** 2 - (predicted ** 2).sum(), 0) * max(total ** 2 - (support ** 2).sum(), 0))
    mcc = (tp.sum() * total - (support * predicted).sum()) / mcc_den if mcc_den > 0 else 0.0

    present = support > 0  # classes absent from the test set must not drag the macro average down
    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1)),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "per_class": {n: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for n, p, r, f, s in zip(names, precision, recall, f1, support)},
        "confusion_matrix": cm.tolist(),
        "class_names": names,
    }


@torch.no_grad()
def predict(model, loader, device):
    """Return ``(y_true, y_pred, probs)`` as numpy arrays for a DataLoader."""
    was_training = model.training
    model.eval()
    model.to(device)
    y_true, y_pred, probs = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        y_true.append(y.cpu())
        y_pred.append(out.argmax(dim=1).cpu())
        probs.append(torch.softmax(out, dim=1).cpu())
    model.train(was_training)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), torch.cat(probs).numpy()


def evaluate_model(model, loader, device, num_classes: int, criterion=None,
                   class_names: Optional[Sequence[str]] = None) -> dict:
    """Full utility report for one model on one DataLoader (e.g. the global held-out test set)."""
    y_true, y_pred, probs = predict(model, loader, device)
    report = classification_report(y_true, y_pred, num_classes, class_names)
    add_probability_metrics(report, probs, y_true)
    if criterion is not None:
        losses = []
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                losses.append(criterion(model(x.to(device)), y.to(device)).item())
        report["loss"] = float(np.mean(losses))
    return report


# ----------------------------------------------------------------------------------------
# extra utility metrics: probabilities, calibration, generalization, client fairness
# ----------------------------------------------------------------------------------------


def calibration_metrics(probs, y_true, n_bins: int = 15) -> dict:
    """
    Expected calibration error, Brier score and NLL, plus the reliability-diagram data.

    Over-confident models (low loss on training samples) are exactly what a loss-threshold
    attack exploits, so calibration helps explain *why* a condition leaks.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int).ravel()
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)

    ece, reliability = 0.0, []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
            reliability.append((float(conf[mask].mean()), float(correct[mask].mean()), int(mask.sum())))
    brier = float(((probs - np.eye(probs.shape[1])[y]) ** 2).sum(axis=1).mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())
    return {"ece": float(ece), "brier": brier, "nll": nll, "reliability": reliability}


def macro_ovr_auroc(probs, y_true) -> float:
    """Macro-average one-vs-rest AUROC over the classes that have both positives and negatives."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int).ravel()
    scores = [auroc(probs[y == k, k], probs[y != k, k])
              for k in range(probs.shape[1]) if 0 < (y == k).sum() < len(y)]
    return float(np.mean(scores))


def add_probability_metrics(report: dict, probs, y_true) -> dict:
    """Add macro one-vs-rest AUROC and calibration to a classification report (in place)."""
    report["macro_auroc_ovr"] = macro_ovr_auroc(probs, y_true)
    report["calibration"] = calibration_metrics(probs, y_true)
    return report


def generalization_gap(train_report: Mapping[str, Any], test_report: Mapping[str, Any]) -> dict:
    """Train minus test accuracy / macro-F1. The gap is the main driver of loss-based MIA."""
    return {"acc_gap": train_report["accuracy"] - test_report["accuracy"],
            "f1_gap": train_report["macro_f1"] - test_report["macro_f1"]}


def client_fairness(per_client_accuracy: Mapping[str, float]) -> dict:
    """Worst / best / mean / std of accuracy across clients, and the best-worst gap."""
    v = np.array(list(per_client_accuracy.values()), dtype=float)
    return {"worst_client_acc": float(v.min()), "best_client_acc": float(v.max()),
            "mean_client_acc": float(v.mean()), "std_client_acc": float(v.std()),
            "client_gap": float(v.max() - v.min())}


def evaluate_per_client(model, client_loaders: Mapping[str, Any], device, num_classes: int) -> dict:
    """Evaluate one (global) model on each client's local test loader, plus the fairness summary."""
    reports = {name: evaluate_model(model, loader, device, num_classes)
               for name, loader in client_loaders.items()}
    return {"clients": reports,
            "fairness": client_fairness({n: r["accuracy"] for n, r in reports.items()})}


# ----------------------------------------------------------------------------------------
# extra privacy metrics
# ----------------------------------------------------------------------------------------


def mia_advantage(member_losses, nonmember_losses) -> float:
    """Attacker advantage: max over thresholds of TPR - FPR (0 = no leakage, 1 = total)."""
    fpr, tpr = roc_curve(-np.asarray(member_losses), -np.asarray(nonmember_losses))
    return float(np.max(tpr - fpr))


def _cp_lower(k, n, a):
    return np.where(k == 0, 0.0, _beta.ppf(a, np.maximum(k, 1), n - k + 1))


def _cp_upper(k, n, a):
    return np.where(k == n, 1.0, _beta.ppf(1 - a, k + 1, np.maximum(n - k, 1)))


def empirical_epsilon(member_losses, nonmember_losses, delta: float = 1e-5, alpha: float = 0.05) -> float:
    """
    An (eps, delta)-DP mechanism forces TPR <= e^eps * FPR + delta (and the mirrored bound for
    FNR/TNR). For every threshold we take a conservative Clopper-Pearson bound on TPR and FPR
    and solve for eps; the maximum over thresholds is returned. It is a *lower bound*, so a
    large value proves weak privacy but a small value does not prove strong privacy. Samples
    from one patient are not independent, so treat it as indicative.
    """
    pos, neg = -np.asarray(member_losses, dtype=float), -np.asarray(nonmember_losses, dtype=float)
    fpr, tpr = roc_curve(pos, neg)
    n_p, n_n = len(pos), len(neg)
    tp = np.rint(tpr * n_p).astype(int)
    fp = np.rint(fpr * n_n).astype(int)

    tpr_lo, tpr_hi = _cp_lower(tp, n_p, alpha / 2), _cp_upper(tp, n_p, alpha / 2)
    fpr_lo, fpr_hi = _cp_lower(fp, n_n, alpha / 2), _cp_upper(fp, n_n, alpha / 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = np.log((tpr_lo - delta) / np.maximum(fpr_hi, 1e-12))            # members flagged more often
        e2 = np.log(((1 - fpr_hi) - delta) / np.maximum(1 - tpr_lo, 1e-12))  # non-members kept out
    e = np.r_[e1[np.isfinite(e1)], e2[np.isfinite(e2)]]
    return float(max(e.max(), 0.0)) if len(e) else 0.0


def paired_auroc_difference(losses_a, losses_b, n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """
    Is condition A's MIA AUROC different from B's? Both models must be scored on the *same*
    members and non-members (same order). Resamples the samples jointly, so the pairing is kept.

    Args:
        losses_a, losses_b: ``(member_losses, nonmember_losses)`` for each model.
    Returns:
        dict with ``diff`` (A - B), 95% ``ci`` and a two-sided bootstrap ``p_value``.
    """
    (ma, na), (mb, nb) = [(np.asarray(m, float), np.asarray(n, float)) for m, n in (losses_a, losses_b)]
    if len(ma) != len(mb) or len(na) != len(nb):
        raise ValueError("Paired comparison needs the same members/non-members for both models.")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        i, j = rng.integers(0, len(ma), len(ma)), rng.integers(0, len(na), len(na))
        diffs.append(auroc(-ma[i], -na[j]) - auroc(-mb[i], -nb[j]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(auroc(-ma, -na) - auroc(-mb, -nb)),
            "ci": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
            "p_value": float(min(max(p, 1 / n_bootstrap), 1.0))}


def pairwise_tests(conditions: Mapping[str, Mapping[str, Any]], pairs=None, **kw) -> pd.DataFrame:
    """Paired AUROC tests; by default each condition against the one before it."""
    names = [n for n, c in conditions.items() if "member_losses" in c.get("mia", {})]
    pairs = pairs or list(zip(names[:-1], names[1:]))
    rows = []
    for a, b in pairs:
        ma, mb = conditions[a]["mia"], conditions[b]["mia"]
        try:
            r = paired_auroc_difference((ma["member_losses"], ma["nonmember_losses"]),
                                        (mb["member_losses"], mb["nonmember_losses"]), **kw)
        except ValueError:
            continue
        rows.append({"A": a, "B": b, "auroc_diff_A_minus_B": r["diff"],
                     "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "p_value": r["p_value"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# results table
# ----------------------------------------------------------------------------------------


def results_table(conditions: Mapping[str, Mapping[str, Any]], baseline: Optional[str] = None) -> pd.DataFrame:
    """
    One row per condition, utility and privacy side by side.

    ``conditions[name]`` holds ``utility`` (from evaluate_model), ``mia`` (from
    LossThresholdAttack.run) and optionally ``epsilon``, ``utility_train`` and ``per_client``.
    ``baseline`` (default: "fedavg" if present, else the first condition) is the reference for
    ``f1_retention`` (F1 / baseline F1) and ``leakage_reduction`` (share of the baseline's
    excess AUROC over 0.5 that has been removed).
    """
    rows = []
    for name, c in conditions.items():
        u, m = c.get("utility", {}), c.get("mia", {})
        ci = m.get("auroc_ci") or [np.nan, np.nan]
        cal = u.get("calibration", {})
        row = {
            "condition": name,
            "epsilon": c.get("epsilon"),
            "accuracy": u.get("accuracy"),
            "macro_f1": u.get("macro_f1"),
            "mcc": u.get("mcc"),
            "kappa": u.get("kappa"),
            "macro_auroc_ovr": u.get("macro_auroc_ovr"),
            "ece": cal.get("ece"),
            "brier": cal.get("brier"),
            "nll": cal.get("nll"),
            "mia_auroc": m.get("auroc"),
            "mia_auroc_lo": ci[0],
            "mia_auroc_hi": ci[1],
            "mia_tpr@1%fpr": m.get("tpr_at_fpr_0.01"),
            "mia_balanced_acc": m.get("balanced_accuracy"),
            "mia_threshold_source": m.get("threshold_source"),
            "loss_gap": m.get("loss_gap"),
            "mia_advantage": np.nan,
            "eps_empirical": np.nan,
        }
        if "member_losses" in m:
            row["mia_advantage"] = mia_advantage(m["member_losses"], m["nonmember_losses"])
            row["eps_empirical"] = empirical_epsilon(m["member_losses"], m["nonmember_losses"])
        if "utility_train" in c:
            row.update(generalization_gap(c["utility_train"], u))
        if "per_client" in c:
            row.update(c["per_client"]["fairness"])
        rows.append(row)

    table = pd.DataFrame(rows)
    base_name = baseline or ("fedavg" if "fedavg" in conditions else table["condition"].iloc[0])
    base = table[table["condition"] == base_name].iloc[0]
    table["f1_retention"] = table["macro_f1"] / base["macro_f1"]
    excess = base["mia_auroc"] - 0.5
    table["leakage_reduction"] = (1 - (table["mia_auroc"] - 0.5) / excess) if excess > 0 else np.nan
    return table


# ----------------------------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------------------------


def _colors(names):
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(names)}


def _save(fig, out_dir, filename):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_history(h):
    """Accept a path to the server's JSON or an already loaded dict."""
    if isinstance(h, (str, Path)):
        with open(h) as f:
            h = json.load(f)
    return h.get("metrics_centralized", h)


# ----------------------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------------------


def plot_training_curves(histories: Mapping[str, Any], out_dir, metrics=("test_accuracy", "test_loss")):
    """Global test metrics per communication round, one line per condition."""
    colors = _colors(histories)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for name, h in histories.items():
            series = _load_history(h).get(metric)
            if not series:
                continue
            rounds, values = zip(*series)
            ax.plot(rounds, values, marker="o", ms=3, label=name, color=colors[name])
        ax.set_xlabel("Communication round")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.3)
    axes[0][0].legend(frameon=False)
    fig.suptitle("Global model on held-out test set")
    return _save(fig, out_dir, "training_curves.png")


def plot_roc_curves(conditions: Mapping[str, Mapping[str, Any]], out_dir):
    """
    MIA ROC curves per condition, linear and log-log. The log-log view shows leakage at low
    false-positive rates, which AUROC alone hides. Needs ``return_losses=True`` in attack.run.
    """
    colors = _colors(conditions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, c in conditions.items():
        m = c["mia"]
        if "member_losses" not in m:
            continue
        fpr, tpr = roc_curve(-np.asarray(m["member_losses"]), -np.asarray(m["nonmember_losses"]))
        label = f"{name} (AUROC {m['auroc']:.3f})"
        ax1.plot(fpr, tpr, label=label, color=colors[name])
        ax2.plot(np.clip(fpr, 1e-4, 1), np.clip(tpr, 1e-4, 1), color=colors[name])
    for ax in (ax1, ax2):
        ax.plot([1e-4 if ax is ax2 else 0, 1], [1e-4 if ax is ax2 else 0, 1], "k--", lw=0.8, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.grid(alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax1.set_title("Membership inference ROC")
    ax2.set_title("Log-log (low-FPR regime)")
    ax1.legend(frameon=False, fontsize=8)
    return _save(fig, out_dir, "mia_roc.png")


def plot_loss_histograms(conditions: Mapping[str, Mapping[str, Any]], out_dir):
    """Member vs non-member loss distributions. Overlap means little leakage."""
    items = [(n, c["mia"]) for n, c in conditions.items() if "member_losses" in c["mia"]]
    cols = min(len(items), 3)
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows), squeeze=False)
    for ax, (name, m) in zip(axes.ravel(), items):
        mem, non = np.asarray(m["member_losses"]), np.asarray(m["nonmember_losses"])
        bins = np.linspace(0, np.percentile(np.r_[mem, non], 99), 40)
        ax.hist(mem, bins=bins, alpha=0.55, density=True, label="members")
        ax.hist(non, bins=bins, alpha=0.55, density=True, label="non-members")
        ax.axvline(m["threshold"], color="k", ls=":", lw=1)
        ax.set_title(f"{name}  (AUROC {m['auroc']:.3f})", fontsize=10)
        ax.set_xlabel("per-sample loss")
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    axes[0][0].legend(frameon=False)
    return _save(fig, out_dir, "mia_loss_histograms.png")


def plot_privacy_utility(table: pd.DataFrame, out_dir):
    """
    Bars for utility (macro-F1) and leakage (MIA AUROC with CI) per condition, plus the
    trade-off scatter (x = utility, y = leakage). Bottom-right (high F1, low AUROC) is the goal.
    """
    names = table["condition"].tolist()
    colors = _colors(names)
    x = np.arange(len(names))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.3))

    a1.bar(x, table["macro_f1"], color=[colors[n] for n in names])
    a1.set_ylabel("Macro-F1 (global test set)")
    a1.set_title("Utility")

    err = np.array([table["mia_auroc"] - table["mia_auroc_lo"], table["mia_auroc_hi"] - table["mia_auroc"]])
    a2.bar(x, table["mia_auroc"], yerr=np.nan_to_num(err), capsize=4, color=[colors[n] for n in names])
    a2.axhline(0.5, color="k", ls="--", lw=0.8)
    a2.text(0.01, 0.5, "chance (no leakage)", transform=a2.get_yaxis_transform(), ha="left", va="bottom", fontsize=8)
    a2.set_ylabel("MIA AUROC (95% CI)")
    a2.set_ylim(0.45, max(1.0, float(table["mia_auroc_hi"].max(skipna=True) or 1.0)))
    a2.set_title("Privacy leakage")

    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)

    for n, f1, au in zip(names, table["macro_f1"], table["mia_auroc"]):
        a3.scatter(f1, au, s=90, color=colors[n], zorder=3)
        a3.annotate(n, (f1, au), textcoords="offset points", xytext=(6, 6), fontsize=8)
    a3.axhline(0.5, color="k", ls="--", lw=0.8)
    a3.set_xlabel("Macro-F1 (higher is better)")
    a3.set_ylabel("MIA AUROC (lower is better)")
    a3.set_title("Privacy-utility trade-off")
    a3.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "privacy_utility_tradeoff.png")


def plot_epsilon_sweep(table: pd.DataFrame, out_dir):
    """Utility and leakage against the privacy budget epsilon (only rows that have one)."""
    t = table.dropna(subset=["epsilon"]).sort_values("epsilon")
    if len(t) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t["epsilon"], t["macro_f1"], "o-", label="Macro-F1")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget ε (log scale, smaller = stricter)")
    ax.set_ylabel("Macro-F1")
    ax2 = ax.twinx()
    ax2.errorbar(t["epsilon"], t["mia_auroc"],
                 yerr=np.nan_to_num([t["mia_auroc"] - t["mia_auroc_lo"], t["mia_auroc_hi"] - t["mia_auroc"]]),
                 fmt="s--", color="tab:red", capsize=3, label="MIA AUROC")
    ax2.axhline(0.5, color="tab:red", ls=":", lw=0.8)
    ax2.set_ylabel("MIA AUROC", color="tab:red")
    ax.grid(alpha=0.3)
    fig.legend(loc="upper center", ncol=2, frameon=False)
    return _save(fig, out_dir, "epsilon_sweep.png")


def plot_confusion_matrices(conditions: Mapping[str, Mapping[str, Any]], out_dir):
    """Row-normalised confusion matrix per condition (shows which classes DP hurts)."""
    items = [(n, c["utility"]) for n, c in conditions.items() if "confusion_matrix" in c.get("utility", {})]
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 4), squeeze=False)
    for ax, (name, u) in zip(axes[0], items):
        cm = np.asarray(u["confusion_matrix"], dtype=float)
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        k = len(cm)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(u["class_names"], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(u["class_names"], fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name, fontsize=10)
    return _save(fig, out_dir, "confusion_matrices.png")


def plot_per_class_f1(conditions: Mapping[str, Mapping[str, Any]], out_dir):
    """Grouped bars of per-class F1. DP often hurts rare classes first."""
    names = list(conditions)
    classes = conditions[names[0]]["utility"]["class_names"]
    colors = _colors(names)
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 3, 4))
    for i, n in enumerate(names):
        f1 = [conditions[n]["utility"]["per_class"][c]["f1"] for c in classes]
        ax.bar(np.arange(len(classes)) + i * w, f1, w, label=n, color=colors[n])
    ax.set_xticks(np.arange(len(classes)) + 0.4 - w / 2)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylabel("F1")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, "per_class_f1.png")


def plot_per_client_auroc(per_client: Mapping[str, Mapping[str, Any]], out_dir, title="MIA AUROC per client"):
    """
    Output of ``LossThresholdAttack.run_per_client``. Large differences between clients mean
    leakage is not uniform (e.g. one hospital's data is memorised more than another's).
    """
    names = list(per_client)
    au = np.array([per_client[n]["auroc"] for n in names])
    ci = [per_client[n].get("auroc_ci") or [np.nan, np.nan] for n in names]
    err = np.nan_to_num(np.array([au - [c[0] for c in ci], [c[1] for c in ci] - au]))
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(names), 4))
    ax.bar(names, au, yerr=err, capsize=3, color="tab:purple")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("MIA AUROC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, "mia_per_client.png")


def plot_mia_over_rounds(round_results: Mapping[str, Mapping[str, Any]], out_dir):
    """
    Output of ``LossThresholdAttack.run_checkpoints`` on per-round checkpoints
    (keys like ``round_1`` ...). Shows how leakage grows with training.
    """
    keys = sorted(round_results, key=lambda k: int(k.split("_")[-1]))
    rounds = [int(k.split("_")[-1]) for k in keys]
    au = np.array([round_results[k]["auroc"] for k in keys])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, au, "o-")
    if all(round_results[k].get("auroc_ci") for k in keys):
        lo, hi = zip(*[round_results[k]["auroc_ci"] for k in keys])
        ax.fill_between(rounds, lo, hi, alpha=0.2)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("MIA AUROC")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "mia_over_rounds.png")


def plot_reliability_diagrams(conditions: Mapping[str, Mapping[str, Any]], out_dir):
    """Confidence vs accuracy per condition. Points below the diagonal = over-confident."""
    items = [(n, c["utility"]["calibration"]) for n, c in conditions.items()
             if "calibration" in c.get("utility", {})]
    colors = _colors(conditions)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for name, cal in items:
        conf, acc, _ = zip(*cal["reliability"])
        ax.plot(conf, acc, "o-", ms=3, color=colors[name], label=f"{name} (ECE {cal['ece']:.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "reliability_diagram.png")


def plot_generalization_vs_leakage(table: pd.DataFrame, out_dir):
    """Generalization gap against MIA AUROC: does the attack mostly measure overfitting?"""
    t = table.dropna(subset=["f1_gap"])
    if t.empty:
        return None
    colors = _colors(table["condition"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for _, r in t.iterrows():
        ax.scatter(r["f1_gap"], r["mia_auroc"], s=90, color=colors[r["condition"]])
        ax.annotate(r["condition"], (r["f1_gap"], r["mia_auroc"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Generalization gap (train F1 - test F1)")
    ax.set_ylabel("MIA AUROC")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "generalization_vs_leakage.png")


def plot_empirical_epsilon(table: pd.DataFrame, out_dir):
    """Attack-implied epsilon lower bound per condition, with the theoretical epsilon marked."""
    t = table.dropna(subset=["eps_empirical"])
    if t.empty:
        return None
    colors = _colors(table["condition"])
    fig, ax = plt.subplots(figsize=(1.5 + 1.3 * len(t), 4))
    ax.bar(t["condition"], t["eps_empirical"], color=[colors[n] for n in t["condition"]])
    theo = t.dropna(subset=["epsilon"])
    ax.scatter(theo["condition"], theo["epsilon"], marker="_", s=900, color="k", zorder=3,
               label="theoretical ε")
    ax.set_ylabel("Empirical ε lower bound")
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(t["condition"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    if len(theo):
        ax.legend(frameon=False)
    return _save(fig, out_dir, "empirical_epsilon.png")


def plot_client_fairness(conditions: Mapping[str, Mapping[str, Any]], out_dir):
    """Per-client accuracy per condition. Spread = some hospitals are served worse."""
    items = [(n, c["per_client"]["clients"]) for n, c in conditions.items() if "per_client" in c]
    if not items:
        return None
    colors = _colors(conditions)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, clients) in enumerate(items):
        acc = [r["accuracy"] for r in clients.values()]
        ax.scatter(np.full(len(acc), i) + np.linspace(-0.15, 0.15, len(acc)), acc, color=colors[name], s=30)
        ax.hlines(np.mean(acc), i - 0.25, i + 0.25, color="k")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([n for n, _ in items], rotation=25, ha="right")
    ax.set_ylabel("Client test accuracy")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, "client_fairness.png")


# ----------------------------------------------------------------------------------------
# one call for everything
# ----------------------------------------------------------------------------------------


def make_report(conditions: Mapping[str, Mapping[str, Any]], out_dir) -> pd.DataFrame:
    """Write the results table (CSV, Markdown) and every figure that the given data supports."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = results_table(conditions)
    table.to_csv(out_dir / "results_table.csv", index=False)
    (out_dir / "results_table.md").write_text(table.round(4).to_markdown(index=False))

    plot_privacy_utility(table, out_dir)
    plot_epsilon_sweep(table, out_dir)
    plot_generalization_vs_leakage(table, out_dir)
    plot_empirical_epsilon(table, out_dir)
    plot_client_fairness(conditions, out_dir)
    if all("calibration" in c.get("utility", {}) for c in conditions.values()):
        plot_reliability_diagrams(conditions, out_dir)
    if all("utility" in c and "confusion_matrix" in c["utility"] for c in conditions.values()):
        plot_confusion_matrices(conditions, out_dir)
        plot_per_class_f1(conditions, out_dir)
    if any("member_losses" in c.get("mia", {}) for c in conditions.values()):
        plot_roc_curves(conditions, out_dir)
        plot_loss_histograms(conditions, out_dir)
    tests = pairwise_tests(conditions)
    if len(tests):
        tests.to_csv(out_dir / "pairwise_auroc_tests.csv", index=False)
        print("\nPaired AUROC tests (A - B):\n", tests.round(4).to_string(index=False))
    histories = {n: c["history"] for n, c in conditions.items() if c.get("history")}
    if histories:
        plot_training_curves(histories, out_dir)

    # raw numbers without the bulky per-sample losses
    slim = {n: {"utility": {k: v for k, v in c.get("utility", {}).items()},
                "mia": {k: v for k, v in c.get("mia", {}).items() if not k.endswith("_losses")},
                "epsilon": c.get("epsilon")} for n, c in conditions.items()}
    (out_dir / "results.json").write_text(json.dumps(slim, indent=2, default=float))
    return table
