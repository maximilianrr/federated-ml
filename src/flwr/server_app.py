import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.data import load_federated_data
from src.flwr.model import VisionModel
from src.flwr.test import evaluate
from src.flwr.train import get_device

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):
    torch.manual_seed(int(context.run_config["seed"]))
    data = load_federated_data(str(context.run_config["cache-dir"]))
    device = get_device()
    output = Path(str(context.run_config["output-dir"]))
    output.mkdir(parents=True, exist_ok=True)
    model = VisionModel(num_classes=data.num_classes)
    initial_arrays = ArrayRecord(model.state_dict())
    best_score, best_state = float("-inf"), None
    history = []

    def validate(server_round, arrays):
        nonlocal best_score, best_state
        model.load_state_dict(arrays.to_torch_state_dict())
        metrics = evaluate(model, data.centralized_val_loader, device)

        if metrics["evaluated_labels"] != data.num_classes:
            raise ValueError("Validation needs positive and negative examples for every finding")
        
        history.append({"round": server_round, **metrics})

        if metrics["macro_ap"] > best_score:
            best_score = metrics["macro_ap"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "round": server_round,
                "validation_metrics": metrics,
                "data_meta": data.meta,
            }, output / "best_model.pt")

        (output / "validation_history.json").write_text(json.dumps(history, indent=2))
        return MetricRecord(metrics)

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=data.num_clients,
        min_available_nodes=data.num_clients,
        weighted_by_key="num-examples",
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=int(context.run_config["num-server-rounds"]),
        timeout=21600,
        train_config=ConfigRecord({
            "learning-rate": float(context.run_config["learning-rate"]),
            "local-epochs": int(context.run_config["local-epochs"]),
        }),
        evaluate_fn=validate,
    )

    torch.save(result.arrays.to_torch_state_dict(), output / "final_model.pt")

    if best_state is None:
        raise RuntimeError("No validated checkpoint was produced")
    
    model.load_state_dict(best_state)
    metrics = evaluate(model, data.global_test_loader, device)
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
