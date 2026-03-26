"""
exporter.py
-----------
Export utilities:
  - ZIP package: model, feature pipeline, datasets, metrics report, README
  - Prediction results as Excel (.xlsx)
"""

import io
import os
import zipfile
import json
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import joblib


RANDOM_SEED = 42


def _serialize_to_bytes(obj: Any) -> bytes:
    """Serialize a Python object to bytes using joblib."""
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    return buf.read()


def export_model_zip(
    model: Any,
    feature_columns: list,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metrics_dict: dict,
    model_name: str = "model",
    val_df: Optional[pd.DataFrame] = None,
) -> io.BytesIO:
    """
    Create a downloadable ZIP containing:
      - model.pkl
      - feature_columns.json  (ordered list of feature names)
      - X_train.csv, y_train.csv
      - X_test.csv, y_test.csv
      - validation.csv (optional)
      - metrics_report.csv
      - README.txt

    Returns a BytesIO buffer.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Model
        model_bytes = _serialize_to_bytes(model)
        zf.writestr(f"{model_name}.pkl", model_bytes)

        # 2. Feature column list (the pipeline "contract")
        zf.writestr("feature_columns.json", json.dumps(feature_columns, indent=2))

        # 3. Train/test datasets
        zf.writestr("X_train.csv", X_train.to_csv(index=False))
        zf.writestr("y_train.csv", y_train.to_frame(name="Activity").to_csv(index=False))
        zf.writestr("X_test.csv", X_test.to_csv(index=False))
        zf.writestr("y_test.csv", y_test.to_frame(name="Activity").to_csv(index=False))

        # 4. Validation set (if provided)
        if val_df is not None:
            zf.writestr("validation.csv", val_df.to_csv(index=False))

        # 5. Metrics report
        metrics_df = pd.DataFrame([metrics_dict])
        zf.writestr("metrics_report.csv", metrics_df.to_csv(index=False))

        # 6. README
        readme_content = f"""Classification Model Builder — Export Package
================================================
Generated     : {timestamp}
Model         : {model_name}

Contents
--------
{model_name}.pkl           - Trained model (joblib format)
feature_columns.json     - Ordered list of feature names used during training
X_train.csv / y_train.csv - Training data
X_test.csv  / y_test.csv  - Test data
metrics_report.csv       - Final evaluation metrics
validation.csv           - Validation set (if available)

How to Load the Model
---------------------
    import joblib, json, pandas as pd
    model = joblib.load('{model_name}.pkl')
    feature_cols = json.load(open('feature_columns.json'))
    # Align your feature DataFrame to feature_cols before predicting:
    X_new = X_new[feature_cols]
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]

Reproducibility
---------------
Random seed: {RANDOM_SEED}
"""
        zf.writestr("README.txt", readme_content)

    zip_buf.seek(0)
    return zip_buf


def export_predictions_xlsx(
    smiles_list: list,
    y_pred: list,
    y_proba_0: Optional[list] = None,
    y_proba_1: Optional[list] = None,
    extra_cols: Optional[dict] = None,
) -> io.BytesIO:
    """
    Export prediction results to an Excel file.

    Returns a BytesIO buffer.
    """
    df = pd.DataFrame({"SMILES": smiles_list, "Prediction": y_pred})
    if y_proba_0 is not None:
        df["Probability_Class0"] = [round(p, 4) for p in y_proba_0]
    if y_proba_1 is not None:
        df["Probability_Class1"] = [round(p, 4) for p in y_proba_1]
    df["Predicted_Label"] = df["Prediction"].map({0: "Inactive", 1: "Active"})
    if extra_cols:
        for col_name, values in extra_cols.items():
            df[col_name] = values

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    buf.seek(0)
    return buf
