"""
Loss-threshold membership inference attack (MIA).

The attacker guesses that a sample was part of the training data ("member") when the model's
loss on it is low, and that it was not ("non-member") when the loss is high. How well this
works is the leakage measurement: AUROC close to 0.5 means little leakage, close to 1.0 means
strong leakage.

This module is deliberately lenient about its inputs, because the dataset, the model and the
training pipeline are not final yet. It only needs per-sample losses, and can get them from:

* model: an ``nn.Module``, any callable returning logits (tensor / array / tuple / object with
  ``.logits``), or a path or dict holding a state_dict (together with ``model_factory``).
* data: a DataLoader or any iterable of batches (``(x, y)`` or dicts), a torch Dataset, a list
  of ``(x, y)`` samples, a tuple ``(X, y)``, or precomputed values: a 1-D array of losses,
  ``{"losses": ...}`` or ``{"logits": ..., "labels": ...}``.

Example:
    attack = LossThresholdAttack(n_bootstrap=200)
    results = attack.run(members, nonmembers, model=model, calibration=(cal_members, cal_nonmembers))
    save_results(results, "outputs/mia_fedavg.json")
"""

import contextlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

# keys that are recognised when a batch is a dict (e.g. HuggingFace / flwr-datasets style)
_X_KEYS = ("image", "img", "x", "input", "inputs", "data", "pixel_values")
_Y_KEYS = ("label", "labels", "y", "target", "targets")
_EPS = 1e-12


# ----------------------------------------------------------------------------------------
# metrics (pure numpy, no sklearn needed)
# ----------------------------------------------------------------------------------------

def auroc(member_scores, nonmember_scores) -> float:
    """
    Area under the ROC curve when members should get higher scores than non-members.

    Ties count as half, so identical score distributions give exactly 0.5.

    Args:
        member_scores (array-like): Scores of the member samples.
        nonmember_scores (array-like): Scores of the non-member samples.
    Returns:
        float: The AUROC between 0 and 1.
    """
    pos = np.asarray(member_scores, dtype=float).ravel()
    neg = np.asarray(nonmember_scores, dtype=float).ravel()
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUROC needs at least one member and one non-member.")

    scores = np.concatenate([pos, neg])
    # average ranks, so tied scores share their rank
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2
    ranks = avg_rank[inverse]

    rank_sum = ranks[: len(pos)].sum()
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc_curve(member_scores, nonmember_scores):
    """
    False and true positive rates over all thresholds (members are the positive class).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(fpr, tpr)``, both starting at 0 and ending at 1.
    """
    pos = np.asarray(member_scores, dtype=float).ravel()
    neg = np.asarray(nonmember_scores, dtype=float).ravel()

    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]

    # only cut the curve where the score changes, so tied scores move it diagonally
    cuts = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    tpr = np.r_[0.0, np.cumsum(labels)[cuts] / len(pos)]
    fpr = np.r_[0.0, np.cumsum(1 - labels)[cuts] / len(neg)]
    return fpr, tpr


def tpr_at_fpr(member_scores, nonmember_scores, target_fpr: float) -> float:
    """
    Highest true positive rate that keeps the false positive rate at or below ``target_fpr``.

    AUROC can hide leakage that only shows up at a low false positive rate, so this is
    reported next to it. With few non-members, very small targets (e.g. 0.001) are noisy.
    """
    fpr, tpr = roc_curve(member_scores, nonmember_scores)
    return float(tpr[fpr <= target_fpr].max())


def accuracy_at_threshold(member_losses, nonmember_losses, threshold: float):
    """
    Accuracy of the rule "member if loss <= threshold".

    Returns:
        tuple[float, float]: ``(accuracy, balanced_accuracy)``. The two are equal when there
        are as many members as non-members; the balanced one is the fair number otherwise.
    """
    m = np.asarray(member_losses, dtype=float).ravel()
    n = np.asarray(nonmember_losses, dtype=float).ravel()
    tpr = np.mean(m <= threshold)
    tnr = np.mean(n > threshold)
    accuracy = (np.sum(m <= threshold) + np.sum(n > threshold)) / (len(m) + len(n))
    return float(accuracy), float((tpr + tnr) / 2)


def best_threshold(member_losses, nonmember_losses):
    """
    Loss threshold with the highest balanced accuracy on the given samples.

    Returns:
        tuple[float, float]: ``(threshold, balanced_accuracy)``.
    """
    m = np.sort(np.asarray(member_losses, dtype=float).ravel())
    n = np.sort(np.asarray(nonmember_losses, dtype=float).ravel())

    # -inf means "call everything a non-member"; the rest are the observed loss values
    candidates = np.r_[-np.inf, np.unique(np.concatenate([m, n]))]
    tpr = np.searchsorted(m, candidates, side="right") / len(m)
    tnr = 1 - np.searchsorted(n, candidates, side="right") / len(n)
    balanced = (tpr + tnr) / 2

    best = int(np.argmax(balanced))
    return float(candidates[best]), float(balanced[best])


# ----------------------------------------------------------------------------------------
# helpers for turning "whatever the caller passed" into tensors and losses
# ----------------------------------------------------------------------------------------

def _as_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))


def _split_batch(batch):
    """Get ``(inputs, labels)`` out of a batch that is a tuple, list or dict."""
    if isinstance(batch, Mapping):
        x = next((batch[k] for k in _X_KEYS if k in batch), None)
        y = next((batch[k] for k in _Y_KEYS if k in batch), None)
        if x is None or y is None:
            raise KeyError(f"Batch keys {list(batch)} do not contain an input key "
                           f"{_X_KEYS} and a label key {_Y_KEYS}.")
        return x, y
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]  # anything after the labels (ids, ...) is ignored
    raise TypeError(f"Cannot read a batch of type {type(batch).__name__}; "
                    "expected (inputs, labels) or a dict.")


def _batches(data, batch_size: int):
    """Yield ``(inputs, labels)`` from any of the supported data containers."""
    if isinstance(data, DataLoader):
        iterator = data
    elif isinstance(data, tuple) and len(data) == 2:
        x, y = _as_tensor(data[0]), _as_tensor(data[1])
        if len(x) != len(y):
            raise ValueError(f"X has {len(x)} rows but y has {len(y)}.")
        iterator = ((x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(x), batch_size))
    elif hasattr(data, "__len__") and hasattr(data, "__getitem__"):
        # torch Dataset or list of samples; never shuffled so the order stays reproducible
        iterator = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        iterator = data  # any other iterable of batches

    for batch in iterator:
        yield _split_batch(batch)


@contextlib.contextmanager
def _eval_mode(model):
    """Put an nn.Module in eval mode (dropout off, batch-norm frozen) and restore it after."""
    if not isinstance(model, nn.Module):
        yield
        return
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def load_model(model, model_factory: Optional[Callable[[], nn.Module]] = None, device=None):
    """
    Turn a model, a state_dict or a checkpoint path into something callable.

    Args:
        model: An nn.Module or other callable (returned unchanged), a state_dict, or a path
            to a file written by ``torch.save(state_dict, path)``. Checkpoints that wrap the
            weights under ``model_state_dict`` / ``state_dict`` / ``model`` also work.
        model_factory (callable): Builds an empty model to load the weights into, e.g.
            ``lambda: VisionModel(input_size=1296, num_classes=5)``. Needed for state_dicts.
        device: Where to load the weights. Defaults to the CPU.
    Returns:
        The model, ready to call.
    """
    if not isinstance(model, (str, Path, Mapping)):
        return model
    if model_factory is None:
        raise ValueError("A state_dict or checkpoint path needs a model_factory to load into.")

    state = model
    if isinstance(model, (str, Path)):
        state = torch.load(model, map_location=device or "cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, Mapping) and key in state:
            state = state[key]
            break

    network = model_factory()
    network.load_state_dict(state)
    return network.to(device) if device is not None else network


# ----------------------------------------------------------------------------------------
# the attack
# ----------------------------------------------------------------------------------------

class LossThresholdAttack:
    """
    Loss-threshold membership inference attack with its evaluation.

    Args:
        loss_fn (str | callable | None): How the per-sample loss is computed.
            ``None`` picks automatically: ``"bce"`` (multi-label, or a single logit) when the
            labels have the same shape as the logits, otherwise ``"ce"`` (single-label).
            ``"ce"`` also accepts one-hot labels. A callable ``f(logits, labels)`` must return
            one loss per sample, shape ``(N,)``.
        from_probs (bool): Set to True if the model outputs probabilities instead of logits.
        ignore_label (float | None): Label entries equal to this value (e.g. -1 for the
            "uncertain" CheXpert labels) are left out of the multi-label loss.
        label_reduction (str): How the multi-label loss is combined across labels, ``"mean"``
            or ``"sum"``.
        batch_size (int): Batch size used when the data is not already in a DataLoader.
        device: Device to run the model on. Defaults to the model's own device, else the CPU.
        n_bootstrap (int): Number of bootstrap resamples for the AUROC confidence interval.
            0 skips it.
        fpr_targets (Sequence[float]): False positive rates at which the TPR is reported.
        seed (int): Seed for the bootstrap.
    """

    def __init__(self, loss_fn=None, from_probs=False, ignore_label=None, label_reduction="mean",
                 batch_size=64, device=None, n_bootstrap=0, fpr_targets=(0.01, 0.001), seed=0):
        if label_reduction not in ("mean", "sum"):
            raise ValueError("label_reduction must be 'mean' or 'sum'.")
        if isinstance(loss_fn, str) and loss_fn not in ("ce", "bce"):
            raise ValueError("loss_fn must be None, 'ce', 'bce' or a callable.")

        self.loss_fn = loss_fn
        self.from_probs = from_probs
        self.ignore_label = ignore_label
        self.label_reduction = label_reduction
        self.batch_size = batch_size
        self.device = device
        self.n_bootstrap = n_bootstrap
        self.fpr_targets = tuple(fpr_targets)
        self.seed = seed

    # -- losses ----------------------------------------------------------------------------

    def losses(self, data, model=None, model_factory=None) -> np.ndarray:
        """
        Per-sample loss of ``model`` on ``data``, in the order the data is iterated.

        Args:
            data: Any supported data container (see the module docstring).
            model: Needed unless ``data`` already holds losses or logits.
            model_factory (callable): Only needed if ``model`` is a state_dict or a path.
        Returns:
            np.ndarray: Losses of shape ``(N,)``.
        """
        # precomputed values: no model needed
        if isinstance(data, Mapping):
            if "losses" in data:
                return self._check(np.asarray(_as_tensor(data["losses"]), dtype=float).ravel())
            if "logits" in data and "labels" in data:
                loss = self._loss_from_outputs(_as_tensor(data["logits"]).float(), _as_tensor(data["labels"]))
                return self._check(loss.numpy())
            raise KeyError("A dict of precomputed values needs 'losses' or both 'logits' and 'labels'.")
        if isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 1:
            return self._check(np.asarray(_as_tensor(data), dtype=float))

        if model is None:
            raise ValueError("A model is required to compute losses from raw samples.")
        model = load_model(model, model_factory, self.device)
        device = self._device_of(model)
        if isinstance(model, nn.Module):
            model.to(device)

        all_losses = []
        with _eval_mode(model), torch.no_grad():
            for x, y in _batches(data, self.batch_size):
                logits = self._forward(model, _as_tensor(x).to(device))
                loss = self._loss_from_outputs(logits, _as_tensor(y).to(device))
                all_losses.append(loss.cpu())

        if not all_losses:
            raise ValueError("The data is empty.")
        return self._check(torch.cat(all_losses).numpy())

    def _device_of(self, model):
        if self.device is not None:
            return torch.device(self.device)
        if isinstance(model, nn.Module):
            return next(model.parameters(), torch.empty(0)).device
        return torch.device("cpu")

    @staticmethod
    def _forward(model, x) -> torch.Tensor:
        out = model(x)
        if hasattr(out, "logits"):
            out = out.logits
        elif isinstance(out, (tuple, list)):
            out = out[0]
        return _as_tensor(out).float()

    def _loss_from_outputs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.detach()
        if callable(self.loss_fn):
            return _as_tensor(self.loss_fn(logits, labels)).detach().reshape(-1).float().cpu()

        kind = self.loss_fn
        if kind is None:
            single_logit = logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1)
            kind = "bce" if single_logit or labels.shape == logits.shape else "ce"

        if kind == "ce":
            if logits.ndim == 2 and labels.shape == logits.shape:
                labels = labels.argmax(dim=1)  # one-hot labels
            labels = labels.reshape(-1).long()
            if self.from_probs:
                picked = logits.gather(1, labels[:, None]).squeeze(1)
                loss = -torch.log(picked.clamp_min(_EPS))
            else:
                loss = F.cross_entropy(logits, labels, reduction="none")
            return loss.float().cpu()

        # binary / multi-label
        if logits.ndim == 1:
            logits = logits[:, None]
        labels = labels.reshape(logits.shape)
        valid = torch.ones_like(logits, dtype=torch.bool)
        if self.ignore_label is not None:
            valid = labels != self.ignore_label
        labels = torch.where(valid, labels, torch.zeros_like(labels)).float()

        if self.from_probs:
            p = logits.clamp(_EPS, 1 - _EPS)
            loss = -(labels * torch.log(p) + (1 - labels) * torch.log(1 - p))
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        loss = (loss * valid).sum(dim=1)
        if self.label_reduction == "mean":
            loss = loss / valid.sum(dim=1).clamp_min(1)
        return loss.float().cpu()

    @staticmethod
    def _check(losses: np.ndarray) -> np.ndarray:
        if len(losses) == 0:
            raise ValueError("No samples.")
        bad = ~np.isfinite(losses)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} of {len(losses)} losses are NaN or infinite; "
                             "the model may have diverged, or the labels do not match the outputs.")
        return losses

    # -- attack ----------------------------------------------------------------------------

    def run(self, members, nonmembers, model=None, calibration=None, threshold=None,
            member_groups=None, nonmember_groups=None, model_factory=None, return_losses=False) -> dict:
        """
        Attack ``model`` and measure how well members can be told apart from non-members.

        Args:
            members: Data the model was trained on.
            nonmembers: Data the model has never seen, from patients/clients that are not in
                the training set either (otherwise leakage is overstated).
            model: The model to attack; not needed if both data arguments are precomputed.
            calibration (tuple): ``(calibration_members, calibration_nonmembers)``, a separate
                small set used only to pick the threshold. This is the fair way to get an
                accuracy number.
            threshold (float): Use this loss threshold instead (member if loss <= threshold).
            member_groups, nonmember_groups (array-like): One group id per sample, in the same
                order as the data (e.g. patient ids). Used to resample whole groups in the
                bootstrap, because one patient's samples are not independent. Do not combine
                with a shuffling DataLoader.
            model_factory (callable): Only needed if ``model`` is a state_dict or a path.
            return_losses (bool): Also return the per-sample losses.
        Returns:
            dict: AUROC (and its confidence interval if ``n_bootstrap`` > 0), accuracy and
            balanced accuracy at the chosen threshold, ``threshold_source``, TPR at low FPR,
            the oracle (best possible threshold) accuracies, mean losses and their gap.
            ``threshold_source`` is ``"given"``, ``"calibration"``, or ``"oracle"`` when
            neither was supplied; in that case the accuracy is an optimistic upper bound.
        """
        for data, groups in ((members, member_groups), (nonmembers, nonmember_groups)):
            if groups is not None and isinstance(data, DataLoader) and isinstance(data.sampler, RandomSampler):
                raise ValueError("Group ids need a fixed sample order, but this DataLoader shuffles.")

        if model is not None:
            model = load_model(model, model_factory, self.device)  # load a checkpoint once
        member_losses = self.losses(members, model)
        nonmember_losses = self.losses(nonmembers, model)

        results = {
            "n_members": int(len(member_losses)),
            "n_nonmembers": int(len(nonmember_losses)),
            "mean_member_loss": float(member_losses.mean()),
            "mean_nonmember_loss": float(nonmember_losses.mean()),
        }
        results["loss_gap"] = results["mean_nonmember_loss"] - results["mean_member_loss"]

        # a lower loss means "more likely a member", so the score is the negative loss
        results["auroc"] = auroc(-member_losses, -nonmember_losses)
        results["auroc_ci"] = None
        if self.n_bootstrap > 0:
            results["auroc_ci"] = self._bootstrap_auroc(
                member_losses, nonmember_losses, member_groups, nonmember_groups)
        for target in self.fpr_targets:
            results[f"tpr_at_fpr_{target:g}"] = tpr_at_fpr(-member_losses, -nonmember_losses, target)

        # accuracy: oracle threshold (upper bound) always, and the fair one if we can
        oracle_thr, _ = best_threshold(member_losses, nonmember_losses)
        results["accuracy_oracle"], results["balanced_accuracy_oracle"] = accuracy_at_threshold(
            member_losses, nonmember_losses, oracle_thr)

        if threshold is not None:
            source, chosen = "given", float(threshold)
        elif calibration is not None:
            cal_members, cal_nonmembers = calibration
            chosen, _ = best_threshold(self.losses(cal_members, model), self.losses(cal_nonmembers, model))
            source = "calibration"
        else:
            source, chosen = "oracle", oracle_thr

        results["threshold"] = chosen
        results["threshold_source"] = source
        results["accuracy"], results["balanced_accuracy"] = accuracy_at_threshold(
            member_losses, nonmember_losses, chosen)

        if return_losses:
            results["member_losses"] = member_losses
            results["nonmember_losses"] = nonmember_losses
        return results

    def run_per_client(self, client_members: Mapping[str, Any], nonmembers, model=None,
                       member_groups: Optional[Mapping[str, Any]] = None, calibration=None, **kwargs) -> dict:
        """
        Run the attack once per client: that client's training data against the same non-members.

        Args:
            client_members: Maps a client name to the data that client trained on.
            nonmembers: Non-member data shared by all clients.
            model: The (global) model to attack.
            member_groups: Optional map from client name to group ids, as in ``run``.
            calibration: As in ``run``.
            **kwargs: Passed on to ``run`` (``threshold``, ``nonmember_groups``, ...).
        Returns:
            dict: Client name to the result of ``run``.
        """
        if model is not None:
            model = load_model(model, kwargs.pop("model_factory", None), self.device)
        # the non-members and calibration set are the same for every client, so score them once
        nonmember_losses = {"losses": self.losses(nonmembers, model)}
        if calibration is not None:
            calibration = ({"losses": self.losses(calibration[0], model)},
                           {"losses": self.losses(calibration[1], model)})

        member_groups = member_groups or {}
        return {name: self.run(data, nonmember_losses, model=model, calibration=calibration,
                               member_groups=member_groups.get(name), **kwargs)
                for name, data in client_members.items()}

    def run_checkpoints(self, checkpoints: Mapping[str, Any], members, nonmembers,
                        model_factory=None, **kwargs) -> dict:
        """
        Run the attack on several models, e.g. one per communication round or per condition.

        Args:
            checkpoints: Maps a name (``"round_3"``, ``"fedavg_dp_strict"``, ...) to a model,
                a state_dict or a checkpoint path.
            members, nonmembers: As in ``run``.
            model_factory (callable): Needed if the checkpoints are state_dicts or paths.
            **kwargs: Passed on to ``run`` (``calibration``, ``threshold``, ...).
        Returns:
            dict: Checkpoint name to the result of ``run``.
        """
        return {name: self.run(members, nonmembers, model=checkpoint, model_factory=model_factory, **kwargs)
                for name, checkpoint in checkpoints.items()}

    # -- bootstrap -------------------------------------------------------------------------

    def _bootstrap_auroc(self, member_losses, nonmember_losses, member_groups, nonmember_groups):
        """95% percentile interval for the AUROC, resampling whole groups when given."""
        rng = np.random.default_rng(self.seed)
        member_index = self._group_index(member_groups, len(member_losses))
        nonmember_index = self._group_index(nonmember_groups, len(nonmember_losses))

        def resample(losses, index):
            if index is None:  # no groups: resample single samples
                return losses[rng.integers(0, len(losses), len(losses))]
            picked = rng.integers(0, len(index), len(index))
            return losses[np.concatenate([index[g] for g in picked])]

        values = [auroc(-resample(member_losses, member_index), -resample(nonmember_losses, nonmember_index))
                  for _ in range(self.n_bootstrap)]
        low, high = np.percentile(values, [2.5, 97.5])
        return [float(low), float(high)]

    @staticmethod
    def _group_index(groups, size) -> Optional[list]:
        """For each group, the positions of its samples. None if there are no groups."""
        if groups is None:
            return None
        groups = np.asarray(groups)
        if len(groups) != size:
            raise ValueError(f"Got {len(groups)} group ids for {size} samples.")
        _, inverse = np.unique(groups, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        return np.split(order, np.cumsum(np.bincount(inverse))[:-1])


# ----------------------------------------------------------------------------------------
# saving
# ----------------------------------------------------------------------------------------

def save_results(results: Mapping[str, Any], path) -> None:
    """Write attack results (as returned by the ``run*`` methods) to a JSON file."""

    def convert(value):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            return np.asarray(value).tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialise {type(value).__name__}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
