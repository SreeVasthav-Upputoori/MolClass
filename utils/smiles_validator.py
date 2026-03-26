"""
smiles_validator.py
-------------------
Validates SMILES strings using RDKit.
Returns valid/invalid rows and a detailed validation report.
"""

from rdkit import Chem
import pandas as pd
from typing import Tuple, List


def validate_smiles(smiles: str) -> bool:
    """Return True if SMILES is parseable by RDKit."""
    if not isinstance(smiles, str) or smiles.strip() == "":
        return False
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None
    except Exception:
        return False


def validate_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    activity_col: str = "Activity",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Validate a DataFrame containing SMILES and Activity columns.

    Returns
    -------
    valid_df   : DataFrame with valid SMILES rows
    invalid_df : DataFrame with invalid/missing rows
    report     : dict with summary statistics
    """
    issues: List[dict] = []
    valid_rows: List[int] = []
    invalid_rows: List[int] = []

    for idx, row in df.iterrows():
        row_issues = []

        # Check for missing SMILES
        if pd.isna(row.get(smiles_col, None)) or str(row.get(smiles_col, "")).strip() == "":
            row_issues.append("Missing SMILES")
        elif not validate_smiles(str(row[smiles_col])):
            row_issues.append("Invalid SMILES (RDKit parse failed)")

        # Check for missing Activity
        if pd.isna(row.get(activity_col, None)):
            row_issues.append("Missing Activity value")
        elif str(row[activity_col]).strip() not in ["0", "1", "0.0", "1.0"]:
            row_issues.append(f"Non-binary Activity value: '{row[activity_col]}'")

        if row_issues:
            invalid_rows.append(idx)
            issues.append({"Row": idx, "SMILES": row.get(smiles_col, ""), "Issues": "; ".join(row_issues)})
        else:
            valid_rows.append(idx)

    valid_df = df.loc[valid_rows].copy().reset_index(drop=True)
    invalid_df = pd.DataFrame(issues)

    # Class distribution
    class_dist = {}
    if activity_col in valid_df.columns:
        counts = valid_df[activity_col].astype(int).value_counts().to_dict()
        class_dist = {int(k): int(v) for k, v in counts.items()}
    total = len(df)
    n_valid = len(valid_df)
    n_invalid = len(invalid_df)

    # Imbalance ratio
    imbalance_ratio = None
    if 0 in class_dist and 1 in class_dist and class_dist[1] > 0:
        imbalance_ratio = round(class_dist[0] / class_dist[1], 3)

    report = {
        "total_rows": total,
        "valid_rows": n_valid,
        "invalid_rows": n_invalid,
        "class_distribution": class_dist,
        "imbalance_ratio": imbalance_ratio,
        "duplicate_smiles": int(valid_df[smiles_col].duplicated().sum()) if smiles_col in valid_df.columns else 0,
    }

    return valid_df, invalid_df, report
