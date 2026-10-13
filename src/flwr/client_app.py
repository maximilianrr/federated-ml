import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.data import load_federated_data
from src.flwr.model import VisionModel
from src.flwr.train import get_device, train_local

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    partition_id = int(context.node_config["partition-id"])
    config = msg.content["config"]
    seed = int(context.run_config["seed"]) + 1000 * int(config["server-round"]) + partition_id
    torch.manual_seed(seed)
    data = load_federated_data(
        cache_dir=str(context.run_config["cache-dir"]),
        batch_size=int(context.run_config["batch-size"]),
        seed=seed,
    )
    model = VisionModel(num_classes=data.num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    loader = data.train_loaders[partition_id]
    loss = train_local(
        model, loader, get_device(), data.meta["pos_weight"],
        learning_rate=float(config["learning-rate"]),
        epochs=int(config["local-epochs"]),
    )
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.cpu().state_dict()),
            "metrics": MetricRecord({"train_loss": loss, "num-examples": len(loader.dataset)}),
        }),
        reply_to=msg,
    )
