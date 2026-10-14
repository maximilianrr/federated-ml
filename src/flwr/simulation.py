"""Launch Flower locally in the active environment with visible worker logs."""

from dataclasses import replace

import torch
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from src.flwr.client_app import train
from src.flwr.server_app import main as serve


def run_federated(config, num_clients):
    server, client = ServerApp(), ClientApp()

    @server.main()
    def server_main(grid, context):
        serve(grid, replace(context, run_config=config))

    @client.train()
    def client_train(message, context):
        torch.set_num_threads(2)
        # Keep Flower's run_config immutable while sharing the persistent client state.
        return train(message, replace(context, run_config=config))

    torch.set_num_threads(2)
    print(f"Starting local Flower {config['mode']}: {num_clients} clients, one at a time", flush=True)
    run_simulation(
        server_app=server, client_app=client, num_supernodes=num_clients,
        backend_config={
            "init_args": {"num_cpus": 2, "num_gpus": 0, "include_dashboard": False},
            "client_resources": {"num_cpus": 2, "num_gpus": 0},
        },
    )
