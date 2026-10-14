"""Prepare and load patient-disjoint, multi-label MIMIC-CXR partitions."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Dataset

DEFAULT_CLASSES = ("No Finding", "Cardiomegaly", "Edema", "Atelectasis", "Pleural Effusion")


@dataclass
class PrepConfig:
    mimic_root: str
    cache_dir: str = "data_cache"
    classes: tuple = DEFAULT_CLASSES
    views: tuple = ("PA", "AP")
    image_size: int = 128
    num_clients: int = 5
    global_test_frac: float = 0.10
    mia_nonmember_frac: float = 0.05
    client_val_frac: float = 0.10
    client_test_frac: float = 0.10
    max_patients: int | None = None
    seed: int = 42
    num_workers: int = 4


def _find_file(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern)) or sorted(root.rglob(pattern))

    if not hits:
        raise FileNotFoundError(f"No file matching {pattern!r} under {root}")
    
    return hits[0]


def build_label_table(cfg: PrepConfig) -> pd.DataFrame:
    root, classes = Path(cfg.mimic_root), list(cfg.classes)
    metadata = pd.read_csv(
        _find_file(root, "*metadata.csv*"),
        usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"],
    )
    downloaded = {path.stem for path in (root / "files").rglob("*.jpg")}
    metadata = metadata[metadata["dicom_id"].isin(downloaded)]
    labels = pd.read_csv(_find_file(root, "*chexpert.csv*"))
    missing = set(classes) - set(labels.columns)

    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    
    values = labels[classes]
    if not (values.isin([-1, 0, 1]) | values.isna()).to_numpy().all():
        raise ValueError("Unexpected values in the finding columns")

    df = metadata.merge(
        labels[["subject_id", "study_id", *classes]],
        on=["subject_id", "study_id"], how="inner", validate="many_to_one",
    )
    df = df[df["ViewPosition"].isin(cfg.views)].copy()
    # Blank -> negative; uncertain -> ignored. Keep multiple positive findings.
    df = df[df[classes].fillna(0).ne(-1).any(axis=1)].copy()

    if df.empty:
        raise ValueError(f"No eligible JPG images under {root / 'files'}")

    df["path"] = df.apply(
        lambda r: (
            f"files/p{str(int(r.subject_id))[:2]}/p{int(r.subject_id)}"
            f"/s{int(r.study_id)}/{r.dicom_id}.jpg"
        ), axis=1,
    )
    # Filter the downloaded cohort before patient sampling, splitting, and statistics.
    available = df["path"].map(lambda path: (root / path).is_file())
    print(f"Available frontal images: {int(available.sum()):,}/{len(df):,}")
    df = df.loc[available].copy()
    if cfg.max_patients is not None:
        if cfg.max_patients < 1:
            raise ValueError("max_patients must be positive")
        patients = np.sort(df["subject_id"].unique())
        selected = np.random.default_rng(cfg.seed).choice(
            patients, size=min(cfg.max_patients, len(patients)), replace=False,
        )
        df = df[df["subject_id"].isin(selected)].copy()
    if df.empty:
        raise ValueError(f"No eligible JPG images under {root / 'files'}")
    return df.sort_values(["subject_id", "study_id", "dicom_id"]).reset_index(drop=True)


def assign_splits(df: pd.DataFrame, cfg: PrepConfig) -> pd.DataFrame:
    """
    Patient-level split into global_test / mia_nonmember / client{k}{train,val,test}.
    """
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
    """
    Each patient must live in exactly one (client, split) bucket.
    """

    buckets = df.groupby("subject_id")[["client", "split"]].nunique()
    bad = buckets[(buckets["client"] > 1) | (buckets["split"] > 1)]

    if len(bad):
        raise AssertionError(f"{len(bad)} patients appear in more than one split")


def _load_one(path: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image.draft("L", (size * 4, size * 4))
        image = ImageOps.pad(
            image.convert("L"), (size, size),
            method=Image.Resampling.LANCZOS, color=0,
        )
        return np.asarray(image, dtype=np.uint8)


def prepare_dataset(cfg: PrepConfig) -> None:
    if cfg.image_size < 32 or cfg.num_clients < 1 or cfg.num_workers < 1:
        raise ValueError("Use image_size >= 32 and positive client/worker counts")
    
    for a, b in (
        (cfg.global_test_frac, cfg.mia_nonmember_frac),
        (cfg.client_val_frac, cfg.client_test_frac),
    ):
        if min(a, b) < 0 or a + b >= 1:
            raise ValueError("Split fractions must be nonnegative and sum to less than one")
        
    out = Path(cfg.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("images.npy", "manifest.csv", "meta.json")):
        raise FileExistsError("Choose a new, empty cache directory")

    df = assign_splits(build_label_table(cfg), cfg)
    verify_no_patient_leakage(df)
    for client in range(cfg.num_clients):
        for split in ("train", "val", "test"):
            if not ((df["client"] == client) & (df["split"] == split)).any():
                raise ValueError(f"Client {client} has no {split} images; use more patients")
            
    if not df["split"].eq("global_test").any():
        raise ValueError("The global test set is empty")
    
    values = df.loc[df["split"] == "train", list(cfg.classes)].fillna(0)
    positives, negatives = values.eq(1).sum(), values.eq(0).sum()
    if (positives == 0).any() or (negatives == 0).any():
        raise ValueError("Every finding needs positive and negative training examples")
    
    pos_weight = np.clip(negatives / positives, 1.0, 10.0)

    images = np.lib.format.open_memmap(
        out / "images.npy", mode="w+", dtype=np.uint8,
        shape=(len(df), cfg.image_size, cfg.image_size),
    )
    paths = [str(Path(cfg.mimic_root) / path) for path in df["path"]]
    decode = partial(_load_one, size=cfg.image_size)
    # Bound outstanding work instead of retaining the decoded dataset in RAM.
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        for start in range(0, len(paths), 64):
            for offset, image in enumerate(executor.map(decode, paths[start:start + 64])):
                images[start + offset] = image
            if start % 1024 == 0:
                print(f"Prepared {min(start + 64, len(paths)):,}/{len(paths):,}")

    images.flush()
    del images
    df.drop(columns="path").to_csv(out / "manifest.csv", index=False)
    meta = {
        "schema_version": 2,
        "config": asdict(cfg),
        "classes": list(cfg.classes),
        "num_classes": len(cfg.classes),
        "image_size": cfg.image_size,
        "num_clients": cfg.num_clients,
        "mean": 0.5,
        "std": 0.5,
        "pos_weight": pos_weight.tolist(),
        "label_policy": {"uncertain": "ignore", "blank": "negative"},
    }
    # Written last so a failed preparation cannot be loaded as a complete cache.
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print_summary(df, cfg)


def print_summary(df: pd.DataFrame, cfg: PrepConfig) -> None:
    print("\nImages per split:")
    print(df.groupby("split").size().to_string())
    train = df[df["split"] == "train"]
    values = train[list(cfg.classes)].fillna(0)
    positives = values.eq(1).groupby(train["client"]).sum()
    supervised = values.ne(-1).groupby(train["client"]).sum()
    print("\nTraining positive fraction among supervised labels:")
    print(positives.div(supervised.replace(0, np.nan)).round(3).to_string())
    print(f"\nMIA: {int(df['mia_member'].sum())} members, "
          f"{int(df['split'].eq('mia_nonmember').sum())} non-members")


class XrayDataset(Dataset):
    """Return a normalized image, binary targets, and an uncertainty mask."""

    def __init__(self, images, labels, idx, mean, std):
        self.images, self.labels, self.idx = images, labels, np.asarray(idx)
        self.mean, self.std = np.float32(mean), np.float32(std)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        image = (self.images[j].astype(np.float32) / 255.0 - self.mean) / self.std
        target = (self.labels[j] == 1).astype(np.float32)
        mask = (self.labels[j] != -1).astype(np.float32)
        return torch.from_numpy(image[None]), torch.from_numpy(target), torch.from_numpy(mask)


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
    def num_classes(self) -> int:
        return self.meta["num_classes"]

    @property
    def num_clients(self) -> int:
        return self.meta["num_clients"]


def load_federated_data(
    cache_dir: str = "data_cache",
    batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> FederatedData:
    cache = Path(cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    if meta.get("schema_version") != 2:
        raise ValueError("Rebuild the cache using the multi-label pipeline")
    
    df = pd.read_csv(cache / "manifest.csv")
    images = np.load(cache / "images.npy", mmap_mode="r")
    if images.shape != (len(df), meta["image_size"], meta["image_size"]):
        raise ValueError("Image cache does not match the manifest")

    labels = df[meta["classes"]].fillna(0).to_numpy(dtype=np.float32)
    mean, std = meta["mean"], meta["std"]

    def ds(mask) -> XrayDataset:
        return XrayDataset(images, labels, np.flatnonzero(np.asarray(mask)), mean, std)

    def dl(dataset, bs, shuffle) -> DataLoader:
        g = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=g,
                          pin_memory=False, drop_last=False)

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


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--mimic-root", required=True)
    prep.add_argument("--cache-dir", default="data_cache")
    prep.add_argument("--num-clients", type=int, default=5)
    prep.add_argument("--image-size", type=int, default=128)
    prep.add_argument("--max-patients", type=int)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--num-workers", type=int, default=4)
    info = commands.add_parser("info")
    info.add_argument("--cache-dir", default="data_cache")
    args = vars(parser.parse_args())
    if args.pop("cmd") == "prepare":
        prepare_dataset(PrepConfig(**args))
    else:
        cache = Path(args["cache_dir"])
        meta = json.loads((cache / "meta.json").read_text())
        print_summary(pd.read_csv(cache / "manifest.csv"), PrepConfig(**meta["config"]))


if __name__ == "__main__":
    _cli()
