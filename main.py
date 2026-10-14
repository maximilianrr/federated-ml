"""Run centralized, FedAvg, or image-DP experiments on a shared prepared cohort."""

import argparse
import json
from pathlib import Path
import tomllib

from src.experiments import MODES, train_centralized
from src.data import load_federated_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, help="Override the configured round count")
    parser.add_argument("--mode", choices=(*MODES, "all"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    config["mode"] = args.mode or config["mode"]

    if args.rounds is not None:
        config["num-server-rounds"] = args.rounds

    if int(config["num-server-rounds"]) < 1:
        parser.error("--rounds must be positive")

    modes = MODES if config["mode"] == "all" else (config["mode"],)

    if any(mode.startswith("dp_") for mode in modes):
        if not 0 < config["dp-high-epsilon"] < config["dp-moderate-epsilon"]:
            parser.error("Require 0 < dp-high-epsilon < dp-moderate-epsilon")

    for key in ("cache-dir", "output-dir"):
        config[key] = str((root / config[key]).resolve())

    cache = Path(config["cache-dir"])
    if not (cache / "meta.json").exists():
        parser.error(f"Prepare the dataset first: no meta.json in {cache}")
        
    meta = json.loads((cache / "meta.json").read_text())

    for mode in modes:
        config["mode"] = mode
        if mode == "centralized":
            data = load_federated_data(
                config["cache-dir"], batch_size=int(config["batch-size"]), seed=int(config["seed"]),
            )
            train_centralized(config.copy(), data)
        else:
            from src.flwr.simulation import run_federated

            run_federated(config.copy(), int(meta["num_clients"]))


if __name__ == "__main__":
    main()
