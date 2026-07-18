#!/usr/bin/env python3
"""Train and evaluate a customer-disjoint CatBoost product ranker."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class PreparedSplit:
    frame: pd.DataFrame
    pool: Pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CatBoostRanker using customer-disjoint train, validation, "
            "and test splits."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/catboost_training.json"),
        help="Path to the JSON training configuration.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable_artifact_path(path: Path) -> str:
    """Return a project-relative artifact path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    required_sections = {
        "data",
        "split",
        "features",
        "model",
        "evaluation",
        "outputs",
    }
    missing_sections = sorted(required_sections - set(config))
    if missing_sections:
        raise ValueError(f"Missing config sections: {missing_sections}")
    return config


def validate_split_config(split_config: dict[str, Any]) -> dict[str, float]:
    fractions = {
        "train": float(split_config["train_fraction"]),
        "validation": float(split_config["validation_fraction"]),
        "test": float(split_config["test_fraction"]),
    }
    if any(fraction <= 0 for fraction in fractions.values()):
        raise ValueError("Every split fraction must be positive.")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"Split fractions must sum to 1.0: {fractions}")
    return fractions


def load_training_data(config: dict[str, Any]) -> pd.DataFrame:
    training_path = resolve_path(config["data"]["training_path"])
    metadata_path = resolve_path(config["data"]["group_metadata_path"])

    training_data = pd.read_csv(
        training_path,
        dtype={
            "customer_id": "string",
            "product_id": "string",
            "product_category": "string",
            "business_line": "string",
        },
    )
    group_metadata = pd.read_csv(
        metadata_path,
        dtype={"customer_id": "string"},
        parse_dates=["scoring_date"],
    )

    required_training_columns = {
        "group_id",
        "customer_id",
        "product_id",
        "label",
    }
    required_metadata_columns = {"group_id", "customer_id", "scoring_date"}
    missing_training = sorted(
        required_training_columns - set(training_data.columns)
    )
    missing_metadata = sorted(
        required_metadata_columns - set(group_metadata.columns)
    )
    if missing_training:
        raise ValueError(f"Missing training columns: {missing_training}")
    if missing_metadata:
        raise ValueError(f"Missing metadata columns: {missing_metadata}")
    if group_metadata["group_id"].duplicated().any():
        raise ValueError("Group metadata contains duplicate group IDs.")

    merged = training_data.merge(
        group_metadata,
        on=["group_id", "customer_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["scoring_date"].isna().any():
        raise ValueError("Some training rows have no matching group metadata.")
    if not set(merged["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    group_customer_counts = merged.groupby("group_id")["customer_id"].nunique()
    if not group_customer_counts.eq(1).all():
        raise ValueError("Every ranking group must belong to one customer.")

    group_labels = merged.groupby("group_id")["label"].agg(
        group_size="size",
        positive_count="sum",
    )
    if not group_labels["positive_count"].gt(0).all():
        raise ValueError("Every ranking group must contain a positive row.")
    if not (
        group_labels["positive_count"] < group_labels["group_size"]
    ).all():
        raise ValueError("Every ranking group must contain a negative row.")

    return merged.sort_values(["group_id", "product_id"]).reset_index(drop=True)


def calculate_customer_capacities(
    customer_count: int,
    fractions: dict[str, float],
) -> dict[str, int]:
    train_count = round(customer_count * fractions["train"])
    validation_count = round(customer_count * fractions["validation"])
    test_count = customer_count - train_count - validation_count
    capacities = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    if any(count <= 0 for count in capacities.values()):
        raise ValueError(f"Customer split would be empty: {capacities}")
    return capacities


def assign_customers_to_splits(
    data: pd.DataFrame,
    fractions: dict[str, float],
    random_seed: int,
) -> dict[str, str]:
    customer_stats = (
        data.groupby("customer_id", as_index=False)
        .agg(
            row_count=("label", "size"),
            group_count=("group_id", "nunique"),
            positive_count=("label", "sum"),
        )
    )
    capacities = calculate_customer_capacities(len(customer_stats), fractions)
    target_rows = {
        split_name: len(data) * fractions[split_name]
        for split_name in SPLIT_NAMES
    }
    assigned_rows = {split_name: 0 for split_name in SPLIT_NAMES}
    assigned_customers = {split_name: 0 for split_name in SPLIT_NAMES}

    random_generator = random.Random(random_seed)
    tie_breakers = {
        customer_id: random_generator.random()
        for customer_id in customer_stats["customer_id"]
    }
    customer_stats["_tie_breaker"] = customer_stats["customer_id"].map(
        tie_breakers
    )
    customer_stats = customer_stats.sort_values(
        ["row_count", "_tie_breaker"],
        ascending=[False, True],
    )

    assignments: dict[str, str] = {}
    for customer in customer_stats.itertuples(index=False):
        available_splits = [
            split_name
            for split_name in SPLIT_NAMES
            if assigned_customers[split_name] < capacities[split_name]
        ]
        if not available_splits:
            raise RuntimeError("No split has remaining customer capacity.")

        chosen_split = min(
            available_splits,
            key=lambda split_name: (
                (assigned_rows[split_name] + customer.row_count)
                / target_rows[split_name],
                assigned_customers[split_name] / capacities[split_name],
                SPLIT_NAMES.index(split_name),
            ),
        )
        assignments[str(customer.customer_id)] = chosen_split
        assigned_rows[chosen_split] += int(customer.row_count)
        assigned_customers[chosen_split] += 1

    return assignments


def apply_customer_disjoint_split(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    fractions = validate_split_config(config["split"])
    assignments = assign_customers_to_splits(
        data,
        fractions,
        int(config["split"]["random_seed"]),
    )

    split_labels = data["customer_id"].map(
        lambda customer_id: assignments[str(customer_id)]
    )
    split_frames = {
        split_name: (
            data.loc[split_labels.eq(split_name)]
            .sort_values(["group_id", "product_id"])
            .reset_index(drop=True)
        )
        for split_name in SPLIT_NAMES
    }

    customer_sets = {
        split_name: set(frame["customer_id"])
        for split_name, frame in split_frames.items()
    }
    if customer_sets["train"] & customer_sets["validation"]:
        raise AssertionError("Train and validation customers overlap.")
    if customer_sets["train"] & customer_sets["test"]:
        raise AssertionError("Train and test customers overlap.")
    if customer_sets["validation"] & customer_sets["test"]:
        raise AssertionError("Validation and test customers overlap.")

    split_summary: dict[str, Any] = {}
    for split_name, frame in split_frames.items():
        split_summary[split_name] = {
            "customers": int(frame["customer_id"].nunique()),
            "groups": int(frame["group_id"].nunique()),
            "rows": len(frame),
            "average_candidates_per_group": float(
                len(frame) / frame["group_id"].nunique()
            ),
            "positives": int(frame["label"].sum()),
            "positive_rate": float(frame["label"].mean()),
            "date_min": frame["scoring_date"].min().date().isoformat(),
            "date_max": frame["scoring_date"].max().date().isoformat(),
        }
    split_summary["customer_overlap"] = {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    return split_frames, split_summary


def determine_feature_columns(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    excluded_columns = set(config["features"]["excluded"])
    excluded_columns.add("scoring_date")
    feature_columns = [
        column for column in data.columns if column not in excluded_columns
    ]
    categorical_features = list(config["features"]["categorical"])

    missing_categorical = sorted(
        set(categorical_features) - set(feature_columns)
    )
    if missing_categorical:
        raise ValueError(
            f"Categorical features are not model features: {missing_categorical}"
        )
    if "customer_id" in feature_columns or "group_id" in feature_columns:
        raise AssertionError("Customer and group IDs must not be model features.")
    return feature_columns, categorical_features


def prepare_feature_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    features = frame[feature_columns].copy()
    numeric_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]
    for column in numeric_features:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        ).fillna(0)
    for column in categorical_features:
        features[column] = (
            features[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(object)
        )
    return features


def build_pool(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
) -> Pool:
    features = prepare_feature_frame(
        frame,
        feature_columns,
        categorical_features,
    )
    return Pool(
        data=features,
        label=frame["label"].astype("int8"),
        group_id=frame["group_id"],
        cat_features=categorical_features,
        feature_names=feature_columns,
    )


def prepare_splits(
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    categorical_features: list[str],
) -> dict[str, PreparedSplit]:
    return {
        split_name: PreparedSplit(
            frame=frame,
            pool=build_pool(
                frame,
                feature_columns,
                categorical_features,
            ),
        )
        for split_name, frame in split_frames.items()
    }


def ranking_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    top_k_values: list[int],
) -> tuple[dict[str, float], pd.DataFrame]:
    scored = frame[
        [
            "group_id",
            "customer_id",
            "product_id",
            "previous_paid_purchase_count",
            "label",
        ]
    ].copy()
    scored["prediction"] = predictions
    scored = scored.sort_values(
        ["group_id", "prediction", "product_id"],
        ascending=[True, False, True],
    )
    scored["rank"] = scored.groupby("group_id").cumcount() + 1

    metric_values: dict[str, list[float]] = {
        "mrr": [],
    }
    for top_k in top_k_values:
        metric_values[f"hit_rate@{top_k}"] = []
        metric_values[f"precision@{top_k}"] = []
        metric_values[f"recall@{top_k}"] = []
        metric_values[f"ndcg@{top_k}"] = []

    recommended_products = {
        top_k: set() for top_k in top_k_values
    }
    for _, group in scored.groupby("group_id", sort=False):
        labels = group["label"].to_numpy(dtype=float)
        positive_count = labels.sum()
        if positive_count <= 0:
            raise ValueError("Metric group contains no positive rows.")

        positive_ranks = np.flatnonzero(labels > 0)
        metric_values["mrr"].append(1.0 / float(positive_ranks[0] + 1))

        for top_k in top_k_values:
            top_labels = labels[:top_k]
            hits = float(top_labels.sum())
            metric_values[f"hit_rate@{top_k}"].append(float(hits > 0))
            metric_values[f"precision@{top_k}"].append(
                hits / float(len(top_labels))
            )
            metric_values[f"recall@{top_k}"].append(
                hits / float(positive_count)
            )

            discounts = 1.0 / np.log2(np.arange(len(top_labels)) + 2.0)
            dcg = float((top_labels * discounts).sum())
            ideal_length = min(int(positive_count), top_k)
            ideal_discounts = 1.0 / np.log2(
                np.arange(ideal_length) + 2.0
            )
            ideal_dcg = float(ideal_discounts.sum())
            metric_values[f"ndcg@{top_k}"].append(dcg / ideal_dcg)
            recommended_products[top_k].update(
                group.head(top_k)["product_id"].astype(str)
            )

    metrics = {
        metric_name: float(np.mean(values))
        for metric_name, values in metric_values.items()
    }
    candidate_product_count = scored["product_id"].nunique()
    for top_k in top_k_values:
        metrics[f"catalogue_coverage@{top_k}"] = (
            len(recommended_products[top_k]) / candidate_product_count
        )
        positive_rows = scored["label"].eq(1)
        positive_segments = {
            "first_purchase": (
                positive_rows
                & scored["previous_paid_purchase_count"].eq(0)
            ),
            "repeat_purchase": (
                positive_rows
                & scored["previous_paid_purchase_count"].gt(0)
            ),
        }
        for segment_name, segment_mask in positive_segments.items():
            segment_count = int(segment_mask.sum())
            metrics[f"{segment_name}_recall@{top_k}"] = (
                float(
                    (
                        segment_mask
                        & scored["rank"].le(top_k)
                    ).sum()
                    / segment_count
                )
                if segment_count
                else float("nan")
            )
    return metrics, scored


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def train(config: dict[str, Any]) -> dict[str, Any]:
    data = load_training_data(config)
    split_frames, split_summary = apply_customer_disjoint_split(data, config)
    feature_columns, categorical_features = determine_feature_columns(
        data,
        config,
    )
    prepared = prepare_splits(
        split_frames,
        feature_columns,
        categorical_features,
    )

    print("Customer-disjoint split:")
    for split_name in SPLIT_NAMES:
        print(f"  {split_name}: {split_summary[split_name]}")
    print(f"Model features ({len(feature_columns)}): {feature_columns}")
    print(f"Categorical features: {categorical_features}")

    model = CatBoostRanker(**config["model"])
    model.fit(
        prepared["train"].pool,
        eval_set=prepared["validation"].pool,
        use_best_model=True,
    )

    top_k_values = sorted(
        {int(value) for value in config["evaluation"]["top_k"]}
    )
    evaluation_metrics: dict[str, dict[str, float]] = {}
    baseline_metrics: dict[str, dict[str, dict[str, float]]] = {}
    test_predictions: pd.DataFrame | None = None
    for split_index, split_name in enumerate(("validation", "test")):
        split_frame = prepared[split_name].frame
        predictions = model.predict(prepared[split_name].pool)
        metrics, scored = ranking_metrics(
            split_frame,
            predictions,
            top_k_values,
        )
        evaluation_metrics[split_name] = metrics

        random_generator = np.random.default_rng(
            int(config["split"]["random_seed"]) + split_index
        )
        random_scores = random_generator.random(len(split_frame))
        random_metrics, _ = ranking_metrics(
            split_frame,
            random_scores,
            top_k_values,
        )
        history_scores = (
            split_frame["previous_paid_purchase_count"].fillna(0).to_numpy()
            + (
                np.log1p(
                    split_frame["previous_category_purchase_count"]
                    .fillna(0)
                    .to_numpy()
                )
                * 0.01
            )
            + (random_generator.random(len(split_frame)) * 1e-6)
        )
        history_metrics, _ = ranking_metrics(
            split_frame,
            history_scores,
            top_k_values,
        )
        baseline_metrics[split_name] = {
            "random": random_metrics,
            "purchase_history": history_metrics,
        }
        if split_name == "test":
            test_predictions = scored

    model_path = resolve_path(config["outputs"]["model_path"])
    metrics_path = resolve_path(config["outputs"]["metrics_path"])
    feature_importance_path = resolve_path(
        config["outputs"]["feature_importance_path"]
    )
    predictions_path = resolve_path(
        config["outputs"]["test_predictions_path"]
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.get_feature_importance(
                prepared["train"].pool
            ),
        }
    ).sort_values("importance", ascending=False)
    feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(feature_importance_path, index=False)

    if test_predictions is None:
        raise RuntimeError("Test predictions were not created.")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(predictions_path, index=False)

    results = {
        "split_strategy": "customer_disjoint",
        "split_summary": split_summary,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "best_iteration": int(model.get_best_iteration()),
        "best_score": model.get_best_score(),
        "ranking_metrics": evaluation_metrics,
        "baseline_metrics": baseline_metrics,
        "split_config": config["split"],
        "model_parameters": config["model"],
        "versions": {
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "artifacts": {
            "model": portable_artifact_path(model_path),
            "feature_importance": portable_artifact_path(
                feature_importance_path
            ),
            "test_predictions": portable_artifact_path(predictions_path),
        },
    }
    save_json(metrics_path, results)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved feature importance to {feature_importance_path}")
    print(f"Saved test predictions to {predictions_path}")
    print(
        json.dumps(
            {
                "catboost": evaluation_metrics,
                "baselines": baseline_metrics,
            },
            indent=2,
        )
    )
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
