"""Explicit, configurable execution of the existing data preparation stages."""

from collections.abc import Sequence
from pathlib import Path
import tomllib

import pandas as pd

from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.data import load_purchases
from recommender.scope import DEFAULT_DATASETS_PATH, get_datasets_locs


def run_preparation(
    *, config_path: str | Path = DEFAULT_DATASETS_PATH,
    order: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return prepared tables in configured order; write no output files.

    An explicit ``order`` overrides ``[pipeline].order`` in the TOML config.
    Items are the parsed master catalogue, not scoring-time eligible candidates.
    """
    with Path(config_path).open("rb") as source:
        config = tomllib.load(source)
    if order is None:
        pipeline = config.get("pipeline", {})
        if not isinstance(pipeline, dict):
            raise ValueError("Dataset config requires a [pipeline] table.")
        order = pipeline.get("order")
    if (
        not isinstance(order, (list, tuple)) or not order
        or any(not isinstance(stage, str) for stage in order)
    ):
        raise ValueError("Preparation order must be a nonempty list or tuple of stage names.")
    allowed = {"companies", "items", "purchases"}
    if set(order) - allowed:
        raise ValueError(f"Unknown preparation stages: {sorted(set(order) - allowed)}")
    if len(set(order)) != len(order):
        raise ValueError("Preparation order must not contain duplicate stages.")

    # Validate every required path before starting any stage.
    required = set(order) | ({"purchases", "items"} if {"companies", "purchases"} & set(order) else set())
    paths = {stage: get_datasets_locs(stage, config_path) for stage in sorted(required)}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    tables = {}
    for stage in order:
        if stage == "companies":
            tables[stage] = load_companies(
                paths[stage], config_path=config_path, purchases=tables.get("purchases"),
            )
        elif stage == "items":
            tables[stage] = load_product_catalogue(paths[stage])
        else:
            tables[stage] = load_purchases(paths[stage], items_path=paths["items"])
    return tables
