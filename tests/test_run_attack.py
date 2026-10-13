"""
End-to-end test for src/run_attack.py on a small fake MIMIC-CXR cache.

The cache is built with the real ``assign_splits`` from ``src/data.py``, so the member and
non-member sets are the ones the project actually uses. No MIMIC data or Flower run is needed.
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DEFAULT_CLASSES, PrepConfig, assign_splits, load_federated_data  # noqa: E402
from src.flwr.model import VisionModel  # noqa: E402
from src.flwr.train import masked_bce  # noqa: E402
from src.run_attack import build_attack_sets, load_checkpoint, main  # noqa: E402

IMAGE_SIZE = 32


def write_fake_cache(cache: Path, num_patients=120, images_per_patient=2, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n = num_patients * images_per_patient
    df = pd.DataFrame({
        "dicom_id": [f"d{i}" for i in range(n)],
        "subject_id": np.repeat(np.arange(10_000, 10_000 + num_patients), images_per_patient),
        "study_id": np.arange(n),
        "ViewPosition": "PA",
    })
    for name in DEFAULT_CLASSES:
        df[name] = rng.choice([1.0, 0.0, -1.0, np.nan], size=n, p=[0.3, 0.4, 0.1, 0.2])

    cfg = PrepConfig(mimic_root="unused", cache_dir=str(cache), image_size=IMAGE_SIZE, num_clients=3,
                     mia_nonmember_frac=0.2, seed=seed)
    df = assign_splits(df, cfg)

    cache.mkdir(parents=True)
    np.save(cache / "images.npy", rng.integers(0, 256, (len(df), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
    df.to_csv(cache / "manifest.csv", index=False)
    meta = {
        "schema_version": 2, "config": asdict(cfg), "classes": list(DEFAULT_CLASSES),
        "num_classes": len(DEFAULT_CLASSES), "image_size": IMAGE_SIZE, "num_clients": cfg.num_clients,
        "mean": 0.5, "std": 0.5, "pos_weight": [1.0] * len(DEFAULT_CLASSES),
        "label_policy": {"uncertain": "ignore", "blank": "negative"},
    }
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))
    return json.loads((cache / "meta.json").read_text())


def train_to_memorise(data, epochs=60):
    """Overfit the real model on the hospitals' training data, so membership should leak."""
    torch.manual_seed(0)
    model = VisionModel(num_classes=data.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(epochs):
        for images, targets, masks in data.centralized_train_loader:
            optimizer.zero_grad()
            masked_bce(model(images), targets, masks).mean().backward()
            optimizer.step()
    return model


class RunAttackTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cache = cls.root / "cache"
        cls.meta = write_fake_cache(cls.cache)
        cls.data = load_federated_data(str(cls.cache))

        trained = train_to_memorise(cls.data)
        cls.best = cls.root / "best_model.pt"
        torch.save({"state_dict": trained.state_dict(), "round": 7,
                    "validation_metrics": {"macro_ap": 0.5}, "data_meta": cls.meta}, cls.best)
        cls.untrained = cls.root / "final_model.pt"
        torch.save(VisionModel(num_classes=5).state_dict(), cls.untrained)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_attack_sets_line_up_with_the_manifest(self):
        sets = build_attack_sets(self.data, seed=1)
        df = self.data.manifest

        self.assertEqual(len(sets["members"]), int(df["mia_member"].sum()))
        self.assertEqual(len(sets["members"]), len(sets["nonmembers"]))
        self.assertEqual(sum(len(d) for d in sets["client_members"].values()), len(sets["members"]))

        # members trained, non-members never seen, and no patient on both sides
        member_patients, nonmember_patients = set(sets["member_groups"]), set(sets["nonmember_groups"])
        train_patients = set(df.loc[df["split"] == "train", "subject_id"])
        self.assertTrue(member_patients <= train_patients)
        self.assertFalse(nonmember_patients & train_patients)

        # the calibration set is disjoint from the evaluation set
        cal_members, cal_nonmembers = sets["calibration"]
        self.assertFalse(set(cal_members.idx) & set(sets["members"].idx))
        self.assertTrue(set(df.loc[cal_nonmembers.idx, "split"]) == {"test"})
        self.assertFalse(set(df.loc[cal_members.idx, "mia_member"]) - {False})

    def test_end_to_end_on_an_overfit_and_an_untrained_model(self):
        out = self.root / "mia"
        results = main(["--checkpoint", f"overfit={self.best}", "--checkpoint", f"untrained={self.untrained}",
                        "--cache-dir", str(self.cache), "--output-dir", str(out), "--n-bootstrap", "50"])

        overfit, untrained = results["overfit"], results["untrained"]
        self.assertGreater(overfit["mia"]["auroc"], 0.7)
        self.assertGreater(overfit["mia"]["auroc"], untrained["mia"]["auroc"])
        self.assertEqual(overfit["mia"]["threshold_source"], "calibration")
        self.assertEqual(overfit["checkpoint"], {"path": str(self.best), "round": 7, "cohort_checked": True})
        self.assertFalse(untrained["checkpoint"]["cohort_checked"])
        self.assertEqual(set(overfit["mia_per_client"]), {"hospital_0", "hospital_1", "hospital_2"})

        saved = json.loads((out / "overfit.json").read_text())
        self.assertEqual(len(saved["mia"]["member_losses"]), overfit["mia"]["n_members"])
        self.assertEqual(saved["condition"], "overfit")

    def test_checkpoint_from_another_cohort_is_refused(self):
        other = dict(self.meta, pos_weight=[2.0] * 5)
        path = self.root / "other_cohort.pt"
        torch.save({"state_dict": VisionModel(num_classes=5).state_dict(), "data_meta": other}, path)
        with self.assertRaises(ValueError):
            load_checkpoint(path, self.data)


if __name__ == "__main__":
    unittest.main()
