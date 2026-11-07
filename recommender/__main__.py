"""Run configured preparation with ``python -m recommender``."""

import argparse

from recommender.preparation import run_preparation
from recommender.scope import DEFAULT_DATASETS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets in configured order.")
    parser.add_argument("--config", default=DEFAULT_DATASETS_PATH, help="Dataset TOML config path.")
    parser.add_argument("--order", nargs="+", choices=["companies", "items", "purchases"],
                        help="Override the TOML preparation order.")
    args = parser.parse_args()
    for stage, table in run_preparation(config_path=args.config, order=args.order).items():
        print(f"{stage}: {len(table)} rows")


if __name__ == "__main__":
    main()
