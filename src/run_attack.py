"""
Run the membership inference attack on trained MIMIC-CXR checkpoints.

Usage:
    python -m src.run_attack --checkpoint fedavg=outputs/best_model.pt
    python -m src.run_attack --checkpoint centralized=runs/central/best_model.pt \\
                             --checkpoint fedavg=runs/fedavg/best_model.pt --n-bootstrap 1000

Members and non-members are the sets fixed in ``data.py`` when the cache was prepared:
members are images from the hospitals' training splits, non-members are images from patients
never used in training. Every checkpoint is scored on the same images, so conditions can be
compared directly. The loss is the masked BCE the model is evaluated with (uncertain findings
are ignored, no class weights).

The attack threshold is chosen on a separate calibration set, so the reported attack accuracy
is not tuned on the images it is measured on:
    members:     training images that are not in the evaluation member set
    non-members: images from the hospitals' test splits (patients never used in training)

Each checkpoint gives one JSON file in ``--output-dir`` with the overall result (including the
per-image losses, needed for paired comparisons between conditions) and one result per hospital.
"""

import argparse
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from src.attack_sim import LossThresholdAttack, load_model, save_results
from src.data import FederatedData, XrayDataset, load_federated_data
from src.flwr.model import VisionModel
from src.flwr.train import get_device


def build_attack_sets(data: FederatedData, calibration: bool = True, seed: int = 42) -> dict:
    """
    Collect the attack's member / non-member data and their patient ids from the cache.

    Args:
        data: The loaded cache (``load_federated_data``).
        calibration (bool): Also build the calibration set used to choose the threshold.
        seed (int): Seed for sampling the calibration set.
    Returns:
        dict: ``members`` and ``nonmembers`` (datasets), ``member_groups`` and
        ``nonmember_groups`` (patient ids in the same order), ``client_members`` and
        ``client_groups`` (per hospital), and ``calibration`` (a pair of datasets, or None).
    """
    df = data.manifest
    members = data.mia_member_loader.dataset
    nonmembers = data.mia_nonmember_loader.dataset

    # data.py builds both datasets from the manifest rows in order, so these line up
    member_rows = df[df["mia_member"].astype(bool)]
    nonmember_rows = df[df["split"] == "mia_nonmember"]
    if len(member_rows) != len(members) or len(nonmember_rows) != len(nonmembers):
        raise ValueError("The manifest does not match the attack loaders; rebuild the cache.")

    client_members, client_groups = {}, {}
    for client in range(data.num_clients):
        positions = np.flatnonzero(member_rows["client"].to_numpy() == client)
        client_members[f"hospital_{client}"] = Subset(members, positions)
        client_groups[f"hospital_{client}"] = member_rows["subject_id"].to_numpy()[positions]

    sets = {
        "members": members,
        "nonmembers": nonmembers,
        "member_groups": member_rows["subject_id"].to_numpy(),
        "nonmember_groups": nonmember_rows["subject_id"].to_numpy(),
        "client_members": client_members,
        "client_groups": client_groups,
        "calibration": None,
    }

    if calibration:
        cal_member_idx = np.flatnonzero(((df["split"] == "train") & ~df["mia_member"].astype(bool)).to_numpy())
        cal_nonmember_idx = np.flatnonzero((df["split"] == "test").to_numpy())
        size = min(len(cal_member_idx), len(cal_nonmember_idx), len(nonmembers))
        if size == 0:
            raise ValueError("No images left for a calibration set; run with --no-calibration.")

        rng = np.random.default_rng(seed)
        pick = lambda idx: np.sort(rng.choice(idx, size=size, replace=False))
        make = lambda idx: XrayDataset(members.images, members.labels, idx, members.mean, members.std)
        sets["calibration"] = (make(pick(cal_member_idx)), make(pick(cal_nonmember_idx)))

    return sets


def load_checkpoint(path, data: FederatedData, device=None):
    """
    Load a checkpoint written by ``server_app.py`` and check it belongs to this cohort.

    ``best_model.pt`` stores the cache's ``meta.json``; if it differs from the loaded cache the
    member / non-member labels would be wrong, so this raises. ``final_model.pt`` stores only
    weights and cannot be checked (``cohort_checked`` is False).

    Returns:
        tuple: ``(model, info)`` where ``info`` describes the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    info = {"path": str(path), "round": None, "cohort_checked": False}

    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        if "data_meta" in checkpoint:
            if checkpoint["data_meta"] != data.meta:
                raise ValueError(f"{path} was trained on a different prepared cohort than this cache.")
            info["cohort_checked"] = True
        info["round"] = checkpoint.get("round")

    model = load_model(checkpoint, lambda: VisionModel(num_classes=data.num_classes), device)
    return model, info


def attack_checkpoint(path, data: FederatedData, sets: dict, attack: LossThresholdAttack,
                      device=None, per_client: bool = True) -> dict:
    """Attack one checkpoint: overall result (with losses) and, optionally, per hospital."""
    model, info = load_checkpoint(path, data, device)

    result = {
        "checkpoint": info,
        "mia": attack.run(sets["members"], sets["nonmembers"], model=model,
                          calibration=sets["calibration"],
                          member_groups=sets["member_groups"],
                          nonmember_groups=sets["nonmember_groups"],
                          return_losses=True),
    }
    if per_client:
        result["mia_per_client"] = attack.run_per_client(
            sets["client_members"], sets["nonmembers"], model=model,
            member_groups=sets["client_groups"], nonmember_groups=sets["nonmember_groups"],
            calibration=sets["calibration"])
    return result


def _parse_checkpoint(value: str):
    """``name=path`` or just ``path`` (the name is then the file name without extension)."""
    name, sep, path = value.partition("=")
    if not sep:
        name, path = Path(value).stem, value
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {value!r}")
    return name, path


def main(argv: Optional[list] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True,
                        help="NAME=PATH of a checkpoint to attack; repeat for several conditions")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", default="outputs/mia")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="bootstrap resamples (by patient) for the AUROC interval; 0 to skip")
    parser.add_argument("--no-calibration", action="store_true",
                        help="choose the threshold on the evaluation sets (optimistic accuracy)")
    parser.add_argument("--no-per-client", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    names = [name for name, _ in args.checkpoint]
    if len(set(names)) != len(names):
        parser.error(f"Checkpoint names must be unique: {names}")

    data = load_federated_data(args.cache_dir, seed=args.seed)
    sets = build_attack_sets(data, calibration=not args.no_calibration, seed=args.seed)
    device = get_device()
    attack = LossThresholdAttack(device=device, n_bootstrap=args.n_bootstrap, seed=args.seed)

    output = Path(args.output_dir)
    all_results = {}
    for name, path in args.checkpoint:
        result = {"condition": name, **attack_checkpoint(path, data, sets, attack, device,
                                                         per_client=not args.no_per_client)}
        save_results(result, output / f"{name}.json")
        all_results[name] = result

        mia = result["mia"]
        ci = mia["auroc_ci"]
        ci_text = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        print(f"{name}: AUROC {mia['auroc']:.3f}{ci_text}, "
              f"TPR@1%FPR {mia['tpr_at_fpr_0.01']:.3f}, "
              f"accuracy {mia['accuracy']:.3f} ({mia['threshold_source']} threshold), "
              f"loss gap {mia['loss_gap']:.4f}")

    print(f"\nWrote {len(all_results)} result file(s) to {output}")
    return all_results


if __name__ == "__main__":
    main()
