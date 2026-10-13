"""
Data preparation for the federated MIMIC-CXR project.

Two steps:

1. PREPARE (run once, slow, needs the raw MIMIC-CXR-JPG folder):

       python data.py prepare --mimic-root /path/to/mimic-cxr-jpg --num-clients 5

   - builds a single-label, 5-class table from the CheXpert labels
   - splits at PATIENT level (no patient ever appears in two places):
         global test set | MIA non-member set | N simulated hospitals
         each hospital: train / val / test
   - picks the MIA "member" images (sampled from the hospitals' training data)
   - resizes every selected image to a small grayscale square and caches it as
     one uint8 .npy file + a manifest.csv + meta.json

2. LOAD (fast, called from client_app.py / server_app.py / main.py):

       from data import load_federated_data
       data = load_federated_data("data_cache", batch_size=32)
       data.train_loaders[partition_id]   # per-hospital training loader
       data.global_test_loader            # global held-out test set
       ...

Splits are fixed by the seed and stored in manifest.csv, so every experiment
condition (centralized, FedAvg, FedAvg+DP) sees exactly the same data.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# Five mutually exclusive classes (the model has num_classes=5 + CrossEntropyLoss,
# so the task has to be single-label). Change here if you prefer other findings.
DEFAULT_CLASSES = ("No Finding", "Cardiomegaly", "Edema", "Atelectasis", "Pleural Effusion")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PrepConfig:
    mimic_root: str
    cache_dir: str = "data_cache"
    classes: tuple = DEFAULT_CLASSES
    views: tuple = ("PA", "AP")        # frontal views only; lateral images are dropped
    image_size: int = 36               # VisionModel's fc1 expects 36x36 (input_size=1296)
    num_clients: int = 5               # number of simulated hospitals
    global_test_frac: float = 0.10     # fraction of PATIENTS for the global test set
    mia_nonmember_frac: float = 0.05   # fraction of PATIENTS never used for training (MIA non-members)
    client_val_frac: float = 0.10      # per-hospital fraction of patients for validation
    client_test_frac: float = 0.10     # per-hospital fraction of patients for local test
    max_per_class: Optional[int] = None  # cap images per class (None = use everything)
    seed: int = 42
    num_workers: int = 8               # processes used for image decoding


# --------------------------------------------------------------------------- #
# Step 1: PREPARE
# --------------------------------------------------------------------------- #
def _find_file(root: Path, pattern: str) -> Path:
    """Locate a MIMIC-CXR-JPG csv by glob (file names differ slightly between releases)."""
    hits = sorted(root.glob(pattern)) or sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching '{pattern}' under {root}")
    return hits[0]


def build_label_table(cfg: PrepConfig) -> pd.DataFrame:
    """One row per image: dicom_id, subject_id, study_id, label (0..4)."""
    root = Path(cfg.mimic_root)
    meta = pd.read_csv(
        _find_file(root, "*metadata.csv*"),
        usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"],
    )
    chex = pd.read_csv(_find_file(root, "*chexpert.csv*"))

    classes = list(cfg.classes)
    missing = [c for c in classes if c not in chex.columns]
    if missing:
        raise ValueError(f"Classes not found in the CheXpert file: {missing}")

    # CheXpert coding: 1.0 = positive, 0.0 = negative, -1.0 = uncertain, NaN = not mentioned.
    # Single-label rule: keep a study only if EXACTLY ONE target class is positive and
    # none of the target classes is uncertain. Everything else is ambiguous for a
    # 5-way softmax classifier and is dropped.
    vals = chex[classes]
    keep = ((vals == 1.0).sum(axis=1) == 1) & ((vals == -1.0).sum(axis=1) == 0)
    chex = chex.loc[keep, ["subject_id", "study_id"] + classes].copy()
    chex["label"] = np.argmax((chex[classes].to_numpy() == 1.0), axis=1)

    df = meta.merge(chex[["subject_id", "study_id", "label"]], on=["subject_id", "study_id"], how="inner")
    df = df[df["ViewPosition"].isin(cfg.views)].drop(columns="ViewPosition")

    if cfg.max_per_class is not None:
        parts = [
            g.sample(n=min(len(g), cfg.max_per_class), random_state=cfg.seed)
            for _, g in df.groupby("label")
        ]
        df = pd.concat(parts)

    df = df.sort_values(["subject_id", "study_id", "dicom_id"]).reset_index(drop=True)
    df["path"] = df.apply(
        lambda r: f"files/p{str(r.subject_id)[:2]}/p{r.subject_id}/s{r.study_id}/{r.dicom_id}.jpg", axis=1
    )
    return df


def assign_splits(df: pd.DataFrame, cfg: PrepConfig) -> pd.DataFrame:
    """Patient-level split into global_test / mia_nonmember / client{k}{train,val,test}."""
    rng = np.random.default_rng(cfg.seed)
    patients = np.sort(df["subject_id"].unique())
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(round(n * cfg.global_test_frac))
    n_mia = int(round(n * cfg.mia_nonmember_frac))
    global_test = patients[:n_test]
    mia_nonmember = patients[n_test:n_test + n_mia]
    pool = patients[n_test + n_mia:]

    client_of: dict = {}
    split_of: dict = {}
    for p in global_test:
        client_of[p], split_of[p] = -1, "global_test"
    for p in mia_nonmember:
        client_of[p], split_of[p] = -1, "mia_nonmember"

    # IID: patients randomly and evenly spread over hospitals (array_split -> sizes differ by <= 1)
    for k, pats in enumerate(np.array_split(pool, cfg.num_clients)):
        n_val = max(1, int(round(len(pats) * cfg.client_val_frac)))
        n_te = max(1, int(round(len(pats) * cfg.client_test_frac)))
        for p in pats[:n_te]:
            client_of[p], split_of[p] = k, "test"
        for p in pats[n_te:n_te + n_val]:
            client_of[p], split_of[p] = k, "val"
        for p in pats[n_te + n_val:]:
            client_of[p], split_of[p] = k, "train"

    df = df.copy()
    df["client"] = df["subject_id"].map(client_of).astype(int)
    df["split"] = df["subject_id"].map(split_of)

    # MIA sets: equal-sized, so attack accuracy has a clean 50 % chance level.
    # Members are random images from the hospitals' training data; non-members come from
    # patients that were never used anywhere in training.
    train_idx = df.index[df["split"] == "train"].to_numpy()
    nonmem_idx = df.index[df["split"] == "mia_nonmember"].to_numpy()
    m = min(len(train_idx), len(nonmem_idx))
    members = rng.choice(train_idx, size=m, replace=False)
    keep_nonmem = rng.choice(nonmem_idx, size=m, replace=False)
    df["mia_member"] = False
    df.loc[members, "mia_member"] = True
    df = df.drop(index=np.setdiff1d(nonmem_idx, keep_nonmem)).reset_index(drop=True)
    return df


def verify_no_patient_leakage(df: pd.DataFrame) -> None:
    """Each patient must live in exactly one (client, split) bucket."""
    buckets = df.groupby("subject_id")[["client", "split"]].nunique()
    bad = buckets[(buckets["client"] > 1) | (buckets["split"] > 1)]
    if len(bad):
        raise AssertionError(f"{len(bad)} patients appear in more than one split")


def _load_one(path: str, size: int) -> Optional[np.ndarray]:
    """
    Per-node preprocessing: decode -> grayscale -> square resize -> uint8.
    This is the 'harmonisation' step every simulated hospital applies locally.
    Returns None if the file is missing/corrupt.
    """
    try:
        with Image.open(path) as im:
            im.draft("L", (size * 4, size * 4))  # cheap JPEG downscale while decoding
            im = im.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            return np.asarray(im, dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None


def prepare_dataset(cfg: PrepConfig) -> None:
    out = Path(cfg.cache_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_label_table(cfg)
    print(f"Eligible images: {len(df):,} from {df['subject_id'].nunique():,} patients")
    df = assign_splits(df, cfg)
    verify_no_patient_leakage(df)

    # decode images in parallel
    root = Path(cfg.mimic_root)
    paths = [str(root / p) for p in df["path"]]
    fn = partial(_load_one, size=cfg.image_size)
    arrays = []
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as ex:
        for i, arr in enumerate(ex.map(fn, paths, chunksize=256), start=1):
            arrays.append(arr)
            if i % 5000 == 0:
                print(f"  decoded {i:,}/{len(paths):,}")

    ok = np.array([a is not None for a in arrays])
    if not ok.all():
        print(f"WARNING: {(~ok).sum()} images could not be read and were dropped")
    df = df[ok].reset_index(drop=True)
    images = np.stack([a for a in arrays if a is not None])
    assert len(images) == len(df)

    # normalisation stats from hospitals' TRAINING images only (no val/test/MIA leakage)
    tr = images[(df["split"] == "train").to_numpy()].astype(np.float32) / 255.0
    mean, std = float(tr.mean()), float(tr.std())

    np.save(out / "images.npy", images)
    df.drop(columns="path").to_csv(out / "manifest.csv", index=False)
    meta = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "classes": list(cfg.classes),
        "num_classes": len(cfg.classes),
        "image_size": cfg.image_size,
        "input_size": cfg.image_size ** 2,
        "num_clients": cfg.num_clients,
        "mean": mean,
        "std": std,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print_summary(df, cfg)
    print(f"\nSaved to {out.resolve()}  (images.npy, manifest.csv, meta.json)")


def print_summary(df: pd.DataFrame, cfg: PrepConfig) -> None:
    names = list(cfg.classes)
    print("\nImages per split:")
    print(df.groupby("split").size().to_string())
    print("\nTraining label distribution per hospital (should be similar -> IID):")
    tr = df[df["split"] == "train"]
    tab = pd.crosstab(tr["client"], tr["label"], normalize="index").round(3)
    tab.columns = names
    tab.insert(0, "n_images", tr.groupby("client").size())
    print(tab.to_string())
    print(f"\nMIA sets: {int(df['mia_member'].sum())} members vs "
          f"{int((df['split'] == 'mia_nonmember').sum())} non-members")


# --------------------------------------------------------------------------- #
# Step 2: LOAD
# --------------------------------------------------------------------------- #
class XrayDataset(Dataset):
    """Reads rows `idx` of the cached uint8 array; returns a normalised (1, H, W) float tensor."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, idx: np.ndarray, mean: float, std: float):
        self.images, self.labels, self.idx = images, labels, np.asarray(idx)
        self.mean, self.std = np.float32(mean), np.float32(std)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        x = (self.images[j].astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x[None, :, :]), int(self.labels[j])


@dataclass
class FederatedData:
    # per-hospital (index = Flower partition-id)
    train_loaders: list
    val_loaders: list
    test_loaders: list
    # global
    global_test_loader: DataLoader
    centralized_train_loader: DataLoader   # union of all hospitals' training data (Phase 1 baseline)
    centralized_val_loader: DataLoader
    # membership inference attack (shuffle=False, so scores stay aligned with labels)
    mia_member_loader: DataLoader
    mia_nonmember_loader: DataLoader
    # info
    meta: dict
    manifest: pd.DataFrame

    @property
    def input_size(self) -> int:
        return self.meta["input_size"]

    @property
    def num_classes(self) -> int:
        return self.meta["num_classes"]

    @property
    def num_clients(self) -> int:
        return self.meta["num_clients"]


def load_federated_data(
    cache_dir: str = "data_cache",
    batch_size: int = 32,
    eval_batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> FederatedData:
    cache = Path(cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    df = pd.read_csv(cache / "manifest.csv")
    images = np.load(cache / "images.npy", mmap_mode="r")
    labels = df["label"].to_numpy()
    mean, std = meta["mean"], meta["std"]

    def ds(mask) -> XrayDataset:
        return XrayDataset(images, labels, np.flatnonzero(np.asarray(mask)), mean, std)

    def dl(dataset, bs, shuffle) -> DataLoader:
        g = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=g)

    k = meta["num_clients"]
    in_split = lambda s: (df["split"] == s).to_numpy()
    of_client = lambda c: (df["client"] == c).to_numpy()

    train_ds = [ds(of_client(c) & in_split("train")) for c in range(k)]
    val_ds = [ds(of_client(c) & in_split("val")) for c in range(k)]
    test_ds = [ds(of_client(c) & in_split("test")) for c in range(k)]

    return FederatedData(
        train_loaders=[dl(d, batch_size, True) for d in train_ds],
        val_loaders=[dl(d, eval_batch_size, False) for d in val_ds],
        test_loaders=[dl(d, eval_batch_size, False) for d in test_ds],
        global_test_loader=dl(ds(in_split("global_test")), eval_batch_size, False),
        centralized_train_loader=dl(ConcatDataset(train_ds), batch_size, True),
        centralized_val_loader=dl(ConcatDataset(val_ds), eval_batch_size, False),
        mia_member_loader=dl(ds(df["mia_member"].to_numpy()), eval_batch_size, False),
        mia_nonmember_loader=dl(ds(in_split("mia_nonmember")), eval_batch_size, False),
        meta=meta,
        manifest=df,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli() -> None:
    p = argparse.ArgumentParser(description="MIMIC-CXR federated data preparation")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="build the cached, split dataset")
    prep.add_argument("--mimic-root", required=True, help="folder containing files/ and the csv.gz files")
    prep.add_argument("--cache-dir", default="data_cache")
    prep.add_argument("--num-clients", type=int, default=5)
    prep.add_argument("--image-size", type=int, default=36)
    prep.add_argument("--max-per-class", type=int, default=None)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--num-workers", type=int, default=8)

    info = sub.add_parser("info", help="print a summary of an already prepared cache")
    info.add_argument("--cache-dir", default="data_cache")

    a = p.parse_args()
    if a.cmd == "prepare":
        prepare_dataset(PrepConfig(
            mimic_root=a.mimic_root, cache_dir=a.cache_dir, num_clients=a.num_clients,
            image_size=a.image_size, max_per_class=a.max_per_class, seed=a.seed,
            num_workers=a.num_workers,
        ))
    else:
        meta = json.loads((Path(a.cache_dir) / "meta.json").read_text())
        df = pd.read_csv(Path(a.cache_dir) / "manifest.csv")
        print_summary(df, PrepConfig(mimic_root="", classes=tuple(meta["classes"])))


if __name__ == "__main__":
    _cli()
