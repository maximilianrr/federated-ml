from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.data import load_federated_data
from src.experiments import Experiment

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):
    config = dict(context.run_config)
    data = load_federated_data(str(config["cache-dir"]))
    experiment = Experiment(config, data)

    def validate(server_round, arrays):
        experiment.model.load_state_dict(arrays.to_torch_state_dict())
        return MetricRecord(experiment.validate(server_round))

    def collect_privacy(replies, weight_key):
        if len(replies) != data.num_clients:
            raise RuntimeError("Every DP client must complete each round to preserve accounting continuity")
        
        for reply in replies:
            metrics = reply["metrics"]
            experiment.privacy["clients"][str(metrics["partition-id"])] = {
                key: metrics[key] for key in ("epsilon", "delta", "noise_multiplier", "sample_rate", "steps")
            }

        # Privacy is bounded by the worst client, not a weighted average.
        return MetricRecord({"epsilon": max(
            client["epsilon"] for client in experiment.privacy["clients"].values()
        )})

    train_config = {
        "learning-rate": float(config["learning-rate"]),
        "local-epochs": int(config["local-epochs"]),
    }

    if experiment.privacy is not None:
        train_config.update({
            "target-epsilon": experiment.privacy["target_epsilon"],
            "num-server-rounds": int(config["num-server-rounds"]),
            **{key: config[key] for key in (
                "dp-delta", "dp-max-grad-norm", "dp-physical-batch-size",
            )},
        })

    strategy = FedAvg(
        fraction_train=1.0, fraction_evaluate=0.0,
        min_train_nodes=data.num_clients, min_available_nodes=data.num_clients,
        weighted_by_key="num-examples",
        train_metrics_aggr_fn=collect_privacy if experiment.privacy is not None else None,
    )

    result = strategy.start(
        grid=grid, initial_arrays=ArrayRecord(experiment.model.cpu().state_dict()),
        num_rounds=int(config["num-server-rounds"]), timeout=21600,
        train_config=ConfigRecord(train_config), evaluate_fn=validate,
    )

    if not result.arrays:
        raise RuntimeError("No client updates were aggregated")
    
    experiment.model.load_state_dict(result.arrays.to_torch_state_dict())
    experiment.finish()
