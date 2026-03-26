"""
feature_selector.py
-------------------
Feature selection utilities for the ML pipeline.
Supports:
  - Variance threshold filtering
  - Pearson correlation filtering
  - Model-based selection (Random Forest importance)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

RANDOM_SEED = 42


def variance_threshold_filter(
    X: pd.DataFrame,
    threshold: float = 0.01,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with variance below `threshold`.

    Returns
    -------
    X_filtered    : filtered DataFrame
    removed_cols  : list of removed column names
    """
    # Fill NaN with column medians before computing variance
    X_filled = X.fillna(X.median(numeric_only=True))
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_filled)
    mask = selector.get_support()
    kept_cols = X.columns[mask].tolist()
    removed_cols = X.columns[~mask].tolist()
    return X[kept_cols].copy(), removed_cols


def correlation_filter(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features (Pearson |corr| > threshold).
    Keeps the first of any correlated pair.

    Returns
    -------
    X_filtered   : filtered DataFrame
    removed_cols : list of removed column names
    """
    X_filled = X.fillna(X.median(numeric_only=True))
    corr_matrix = X_filled.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    X_filtered = X.drop(columns=to_drop, errors="ignore")
    return X_filtered, to_drop


def model_based_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 100,
    n_estimators: int = 100,
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Select top-N features by Random Forest feature importances.

    Returns
    -------
    X_selected      : DataFrame with top-N features
    selected_cols   : list of selected column names
    importances     : Series of feature importances (all features)
    """
    X_filled = X.fillna(X.median(numeric_only=True))
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_filled, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_n = min(n_features, len(importances))
    selected_cols = importances.head(top_n).index.tolist()
    return X[selected_cols].copy(), selected_cols, importances


def apply_feature_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    use_variance: bool = True,
    variance_threshold: float = 0.01,
    use_correlation: bool = True,
    correlation_threshold: float = 0.95,
    use_model_based: bool = False,
    n_model_features: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Apply the full feature selection pipeline to train/test sets.
    NOTE: Fitting is always on train set only; test set is transformed accordingly.

    Returns
    -------
    X_train_sel : selected train features
    X_test_sel  : test features aligned to selected train columns
    report      : dict summarising steps and feature counts
    """
    report = {"initial_features": X_train.shape[1], "steps": []}
    X_tr, X_te = X_train.copy(), X_test.copy()

    # Step 1: Drop columns with all-NaN first
    all_nan_cols = [c for c in X_tr.columns if X_tr[c].isna().all()]
    if all_nan_cols:
        X_tr.drop(columns=all_nan_cols, inplace=True)
        X_te = X_te[[c for c in X_te.columns if c in X_tr.columns]]
        report["steps"].append({"name": "Drop all-NaN cols", "removed": len(all_nan_cols), "remaining": X_tr.shape[1]})

    if use_variance:
        X_tr, removed = variance_threshold_filter(X_tr, threshold=variance_threshold)
        X_te = X_te[[c for c in X_te.columns if c in X_tr.columns]]
        report["steps"].append({"name": "Variance filter", "removed": len(removed), "remaining": X_tr.shape[1]})

    if use_correlation:
        X_tr, removed = correlation_filter(X_tr, threshold=correlation_threshold)
        X_te = X_te[[c for c in X_te.columns if c in X_tr.columns]]
        report["steps"].append({"name": "Correlation filter", "removed": len(removed), "remaining": X_tr.shape[1]})

    if use_model_based:
        X_tr, selected_cols, _ = model_based_selection(X_tr, y_train, n_features=n_model_features)
        X_te = X_te[[c for c in selected_cols if c in X_te.columns]]
        report["steps"].append({"name": "RF importance filter", "selected": len(selected_cols), "remaining": X_tr.shape[1]})

    report["final_features"] = X_tr.shape[1]

    # Align test to train columns (fill any missing with 0)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    # Final NaN fill
    col_medians = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(col_medians)
    X_te = X_te.fillna(col_medians)

    return X_tr, X_te, report
