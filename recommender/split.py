"""Chronological dataset partitioning and temporal leakage validation.

Implements the exact 70/15/15 chronological schedule:
- Train: purchase events strictly before 2025-10-01 (< 2025-10-01)
- Validation: purchase events between 2025-10-01 and 2026-02-18 (inclusive)
- Test: purchase events on or after 2026-02-19 (>= 2026-02-19)
"""

from typing import Dict, Tuple
import pandas as pd

TRAIN_CUTOFF = pd.Timestamp("2025-10-01")
VAL_START = pd.Timestamp("2025-10-01")
VAL_END = pd.Timestamp("2026-02-18 23:59:59.999999")
TEST_START = pd.Timestamp("2026-02-19")


def validate_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_column: str = "purchase_date",
) -> bool:
    """Validate that temporal partitions have strict separation and zero future leakage.

    Raises ValueError if any temporal boundary violation, index collision,
    or overlap between splits is detected. Returns True when splits are clean.
    """
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if date_column not in df.columns:
            raise ValueError(f"{name} partition is missing required date column: {date_column}")

    # Check index disjointness
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    if not train_idx.isdisjoint(val_idx):
        overlap = train_idx.intersection(val_idx)
        raise ValueError(f"Temporal leakage: Train and validation share {len(overlap)} indices.")
    if not train_idx.isdisjoint(test_idx):
        overlap = train_idx.intersection(test_idx)
        raise ValueError(f"Temporal leakage: Train and test share {len(overlap)} indices.")
    if not val_idx.isdisjoint(test_idx):
        overlap = val_idx.intersection(test_idx)
        raise ValueError(f"Temporal leakage: Validation and test share {len(overlap)} indices.")

    # Validate dates
    if not train_df.empty:
        train_dates = pd.to_datetime(train_df[date_column])
        if (train_dates >= TRAIN_CUTOFF).any():
            violating = train_df.loc[train_dates >= TRAIN_CUTOFF]
            raise ValueError(
                f"Temporal leakage in train split: {len(violating)} events have date >= {TRAIN_CUTOFF}. "
                f"Max date found: {train_dates.max()}"
            )
        max_train = train_dates.max()
    else:
        max_train = None

    if not val_df.empty:
        val_dates = pd.to_datetime(val_df[date_column])
        if (val_dates < VAL_START).any() or (val_dates >= TEST_START).any():
            violating = val_df.loc[(val_dates < VAL_START) | (val_dates >= TEST_START)]
            raise ValueError(
                f"Temporal leakage in val split: {len(violating)} events fall outside "
                f"[{VAL_START}, {TEST_START}). Min: {val_dates.min()}, Max: {val_dates.max()}"
            )
        min_val = val_dates.min()
        max_val = val_dates.max()
        if max_train is not None and max_train >= min_val:
            raise ValueError(
                f"Temporal boundary inversion: Train max date {max_train} >= Val min date {min_val}"
            )
    else:
        min_val = None
        max_val = None

    if not test_df.empty:
        test_dates = pd.to_datetime(test_df[date_column])
        if (test_dates < TEST_START).any():
            violating = test_df.loc[test_dates < TEST_START]
            raise ValueError(
                f"Temporal leakage in test split: {len(violating)} events have date < {TEST_START}. "
                f"Min date found: {test_dates.min()}"
            )
        min_test = test_dates.min()
        if max_val is not None and max_val >= min_test:
            raise ValueError(
                f"Temporal boundary inversion: Val max date {max_val} >= Test min date {min_test}"
            )
        elif max_train is not None and min_val is None and max_train >= min_test:
            raise ValueError(
                f"Temporal boundary inversion: Train max date {max_train} >= Test min date {min_test}"
            )

    return True


def chronological_split(
    purchases_df: pd.DataFrame,
    date_column: str = "purchase_date",
    validate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset chronologically into Train (70%), Val (15%), Test (15%).

    - Train: purchase_date < 2025-10-01
    - Validation: 2025-10-01 <= purchase_date <= 2026-02-18
    - Test: purchase_date >= 2026-02-19
    """
    if date_column not in purchases_df.columns:
        raise ValueError(f"Input DataFrame is missing required date column: {date_column}")

    dates = pd.to_datetime(purchases_df[date_column])

    train_mask = dates < TRAIN_CUTOFF
    val_mask = (dates >= VAL_START) & (dates < TEST_START)
    test_mask = dates >= TEST_START

    # Verify every row belongs to exactly one partition
    unassigned = ~(train_mask | val_mask | test_mask)
    if unassigned.any():
        unassigned_dates = dates.loc[unassigned]
        raise ValueError(
            f"Unassigned events during split: {unassigned.sum()} events could not be partitioned. "
            f"Sample dates: {unassigned_dates.head().tolist()}"
        )

    train_df = purchases_df.loc[train_mask].copy()
    val_df = purchases_df.loc[val_mask].copy()
    test_df = purchases_df.loc[test_mask].copy()

    if validate:
        validate_no_leakage(train_df, val_df, test_df, date_column=date_column)

    return train_df, val_df, test_df


def get_split_proportions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute relative sample proportions across splits."""
    total = len(train_df) + len(val_df) + len(test_df)
    if total == 0:
        return {"train": 0.0, "val": 0.0, "test": 0.0}
    return {
        "train": len(train_df) / total,
        "val": len(val_df) / total,
        "test": len(test_df) / total,
    }
