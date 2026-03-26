"""
model_trainer.py
----------------
Train and tune multiple binary classification models.
Supported models: RandomForest, LightGBM, XGBoost, SVM, LogisticRegression
Uses RandomizedSearchCV for hyperparameter tuning (5-fold CV).
Random seed: 42 throughout for reproducibility.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, dict] = {
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "param_dist": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "class_weight": [None, "balanced"],
        },
        "needs_scaling": False,
    },
    "LightGBM": {
        "estimator": lgb.LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
        "param_dist": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [-1, 5, 10, 20],
            "num_leaves": [31, 63, 127],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [0.0, 0.1, 0.5],
            "class_weight": [None, "balanced"],
        },
        "needs_scaling": False,
    },
    "XGBoost": {
        "estimator": xgb.XGBClassifier(
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        ),
        "param_dist": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.5],
            "scale_pos_weight": [1, 2, 5],
        },
        "needs_scaling": False,
    },
    "SVM": {
        "estimator": SVC(probability=True, random_state=RANDOM_SEED),
        "param_dist": {
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ["rbf", "linear", "poly"],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"],
        },
        "needs_scaling": True,
    },
    "Logistic Regression": {
        "estimator": LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=2000,
            solver="lbfgs",
        ),
        "param_dist": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "class_weight": [None, "balanced"],
        },
        "needs_scaling": True,
    },
}


def _make_pipeline(model_name: str) -> Tuple[Any, bool]:
    """Build estimator (wrapped in StandardScaler pipeline if needed)."""
    cfg = MODEL_REGISTRY[model_name]
    estimator = cfg["estimator"]
    needs_scaling = cfg["needs_scaling"]

    if needs_scaling:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])
        # Prefix param_dist keys for Pipeline
        param_dist = {f"clf__{k}": v for k, v in cfg["param_dist"].items()}
        return pipeline, param_dist
    return estimator, cfg["param_dist"]


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
    n_iter: int = 20,
    cv_folds: int = 5,
) -> Tuple[Any, dict, dict]:
    """
    Train a model with optional RandomizedSearchCV tuning.

    Parameters
    ----------
    model_name : one of MODEL_REGISTRY keys
    X_train    : feature DataFrame
    y_train    : binary target Series
    tune       : whether to run hyperparameter tuning
    n_iter     : number of random search iterations
    cv_folds   : cross-validation folds

    Returns
    -------
    fitted_model  : fitted estimator (or pipeline)
    best_params   : best hyperparameters found
    cv_results    : cross-validation score summary
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")

    estimator, param_dist = _make_pipeline(model_name)

    # Clean infinity and Fill NaN (safety net)
    X_tr = X_train.replace([np.inf, -np.inf], np.nan)
    X_tr = X_tr.fillna(X_tr.median(numeric_only=True)).fillna(0)

    if tune:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        search.fit(X_tr, y_train)
        fitted_model = search.best_estimator_
        best_params = search.best_params_
        cv_results = {
            "best_cv_auc": round(search.best_score_, 4),
            "mean_test_auc": round(float(np.mean(search.cv_results_["mean_test_score"])), 4),
        }
    else:
        estimator.fit(X_tr, y_train)
        fitted_model = estimator
        best_params = {}
        cv_results = {}

    return fitted_model, best_params, cv_results


def train_all_models(
    model_names: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
    n_iter: int = 20,
    cv_folds: int = 5,
    progress_callback=None,
) -> dict:
    """
    Train multiple models and return a results dict.

    Returns
    -------
    {model_name: {"model": ..., "best_params": ..., "cv_results": ...}}
    """
    results = {}
    for i, name in enumerate(model_names):
        if progress_callback:
            progress_callback(i, len(model_names), name)
        model, params, cv = train_model(name, X_train, y_train, tune=tune, n_iter=n_iter, cv_folds=cv_folds)
        results[name] = {"model": model, "best_params": params, "cv_results": cv}
    return results


def get_predictions(model, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Get class predictions and probability scores from a fitted model.

    Returns
    -------
    y_pred  : array of 0/1 predictions
    y_proba : array of P(class=1) probabilities, or None
    """
    X_cleaned = X.replace([np.inf, -np.inf], np.nan)
    X_filled = X_cleaned.fillna(X_cleaned.median(numeric_only=True)).fillna(0)
    y_pred = model.predict(X_filled)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_filled)[:, 1]
        except Exception:
            pass
    return y_pred, y_proba
