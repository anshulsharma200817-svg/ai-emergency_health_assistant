"""Data Partitioning and Splitting Module.

Educational Simulation System.
Enforces strict 70% train / 15% val / 15% test partitioning with random_seed 42.
Ensures zero patient/trajectory leakage for sequence data.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger

logger = setup_logger("data_splitters")


def split_tabular_data(
    df: pd.DataFrame,
    target_col: str = "risk_class",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split tabular DataFrame into train, validation, and test sets.

    Args:
        df: Input pandas DataFrame.
        target_col: Target column name for stratified splitting.
        train_size: Proportion of training data (0.70).
        val_size: Proportion of validation data (0.15).
        test_size: Proportion of test data (0.15).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of (df_train, df_val, df_test).
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "Split proportions must sum to 1.0"

    if target_col not in df.columns:
        raise ValueError(
            f"target column {target_col!r} is missing; stratified splitting and "
            "training both require it"
        )

    stratify = df[target_col]

    # First split off train set
    df_train, df_temp = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=stratify
    )

    # Calculate ratio for val vs test out of remaining (1 - train_size)
    relative_val_size = val_size / (val_size + test_size)
    stratify_temp = df_temp[target_col]

    df_val, df_test = train_test_split(
        df_temp,
        train_size=relative_val_size,
        random_state=seed,
        stratify=stratify_temp
    )

    logger.info(
        f"Tabular dataset split complete: Train={len(df_train)} (70%), "
        f"Val={len(df_val)} (15%), Test={len(df_test)} (15%)"
    )
    return df_train, df_val, df_test


def split_temporal_data(
    X: np.ndarray,
    y: np.ndarray,
    trajectory_ids: np.ndarray,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split sequence trajectories strictly by trajectory ID to prevent patient data leakage.

    Args:
        X: Sequence data array of shape (N, seq_len, num_features).
        y: Target array of shape (N,).
        trajectory_ids: Array of unique trajectory IDs of shape (N,).
        train_size: Proportion for training set (0.70).
        val_size: Proportion for validation set (0.15).
        test_size: Proportion for test set (0.15).
        seed: Random seed (42).

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
    """
    unique_ids = np.unique(trajectory_ids)

    train_ids, temp_ids = train_test_split(
        unique_ids,
        train_size=train_size,
        random_state=seed
    )

    relative_val_size = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=relative_val_size,
        random_state=seed
    )

    train_mask = np.isin(trajectory_ids, train_ids)
    val_mask = np.isin(trajectory_ids, val_ids)
    test_mask = np.isin(trajectory_ids, test_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(
        f"Temporal sequence split complete: Train trajectories={len(train_ids)}, "
        f"Val trajectories={len(val_ids)}, Test trajectories={len(test_ids)}"
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
