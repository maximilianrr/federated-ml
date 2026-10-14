"""Training orchestration and checkpointing for centralized and federated experiments."""

import copy
import json
from pathlib import Path

import torch

from src.flwr.model import VisionModel
from src.flwr.test import evaluate
from src.flwr.train import get_device, train_local


MODES = ("centralized", "fedavg", "dp_moderate", "dp_high")


class Experiment:
    def __init__(self, config, data):
        self.config, self.data = config, data
        self.mode = str(config["mode"])
        if self.mode not in MODES:
            raise ValueError(f"Unknown training mode: {self.mode}")
        
        torch.manual_seed(int(config["seed"]))
        self.device = get_device()
        self.model = VisionModel(num_classes=data.num_classes).to(self.device)
        self.output = Path(str(config["output-dir"])) / self.mode
        self.output.mkdir(parents=True, exist_ok=True)
        self.best_score, self.history = float("-inf"), []
        self.meta = dict(data.meta)
        self.privacy = None

        if self.mode.startswith("dp_"):
            self.meta.pop("pos_weight", None)
            self.privacy = {
                "unit": "image", "accountant": "integer-order-rdp",
                "target_epsilon": float(config[f"dp-{self.mode[3:]}-epsilon"]),
                "delta": float(config["dp-delta"]),
                "max_grad_norm": float(config["dp-max-grad-norm"]),
                "secure_rng": False, "clients": {},
            }

    def validate(self, step):
        metrics = evaluate(self.model, self.data.centralized_val_loader, self.device)
        if metrics["evaluated_labels"] != self.data.num_classes:
            raise ValueError("Validation needs positive and negative examples for every finding")
        
        if self.privacy is not None:
            metrics["epsilon"] = max(
                (client["epsilon"] for client in self.privacy["clients"].values()), default=0.0,
            )

            (self.output / "privacy.json").write_text(json.dumps(self.privacy, indent=2))

        self.history.append({"round": step, **metrics})
        if metrics["macro_ap"] > self.best_score:
            self.best_score = metrics["macro_ap"]
            torch.save({
                "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "round": step, "validation_metrics": metrics, "data_meta": self.meta,
                "run_config": self.config, "privacy": copy.deepcopy(self.privacy),
            }, self.output / "best_model.pt")

        (self.output / "validation_history.json").write_text(json.dumps(self.history, indent=2))
        return metrics

    def finish(self):
        torch.save(
            {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            self.output / "final_model.pt",
        )
        
        checkpoint = torch.load(self.output / "best_model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        metrics = evaluate(self.model, self.data.global_test_loader, self.device)
        (self.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))


def train_centralized(config, data):
    experiment = Experiment(config, data)
    optimizer = torch.optim.AdamW(
        experiment.model.parameters(), lr=float(config["learning-rate"]), weight_decay=1e-4,
    )
    experiment.validate(0)
    for step in range(1, int(config["num-server-rounds"]) + 1):
        loss = train_local(
            experiment.model, data.centralized_train_loader, experiment.device, data.meta["pos_weight"],
            learning_rate=float(config["learning-rate"]), epochs=int(config["local-epochs"]),
            optimizer=optimizer,
        )
        metrics = experiment.validate(step)
        print(f"Epoch group {step}: train_loss={loss:.4f}, macro_ap={metrics['macro_ap']:.4f}")
    experiment.finish()
