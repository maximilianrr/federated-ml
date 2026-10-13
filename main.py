"""Launch a local Flower simulation using project-relative data and output paths."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, help="Override the configured round count")
    args = parser.parse_args()
    if args.rounds is not None and args.rounds < 1:
        parser.error("--rounds must be positive")
    root = Path(__file__).resolve().parent
    config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    cache = (root / config["cache-dir"]).resolve()
    if not (cache / "meta.json").exists():
        parser.error(f"Prepare the dataset first: no meta.json in {cache}")
    meta = json.loads((cache / "meta.json").read_text())
    # Flower packages the app; pass absolute paths to keep data outside the package.
    overrides = (
        f'cache-dir={json.dumps(str(cache))} '
        f'output-dir={json.dumps(str((root / config["output-dir"]).resolve()))}'
    )
    if args.rounds is not None:
        overrides += f" num-server-rounds={args.rounds}"
    subprocess.run([
        str(Path(sys.executable).with_name("flwr")), "run", str(root),
        "--stream",
        "--run-config", overrides,
        "--federation-config",
        f'num-supernodes={meta["num_clients"]} init-args-num-cpus=2 '
        "init-args-num-gpus=0 client-resources-num-cpus=2 client-resources-num-gpus=0",
    ], cwd=root, check=True)


if __name__ == "__main__":
    main()
