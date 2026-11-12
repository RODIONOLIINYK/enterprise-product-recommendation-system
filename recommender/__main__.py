"""Run configured preparation with ``python -m recommender``."""

import argparse

from recommender.preparation import export_purchases, run_preparation
from recommender.scope import DEFAULT_DATASETS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets in configured order.")
    parser.add_argument("--config", default=DEFAULT_DATASETS_PATH, help="Dataset TOML config path.")
    parser.add_argument("--order", nargs="+", choices=["companies", "items", "purchases"],
                        help="Override the TOML preparation order.")
    parser.add_argument("--export-purchases", metavar="PATH", help="Explicitly write the prepared purchases CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of the explicitly chosen export.")
    args = parser.parse_args()
    tables = run_preparation(config_path=args.config, order=args.order)
    for stage, table in tables.items():
        print(f"{stage}: {len(table)} rows")
    if args.export_purchases:
        if "purchases" not in tables:
            parser.error("Export requires the purchases preparation stage")
        export_purchases(tables["purchases"], args.export_purchases, overwrite=args.overwrite)
        print(f"Purchases exported to {args.export_purchases}")


if __name__ == "__main__":
    main()
