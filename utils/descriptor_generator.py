"""
descriptor_generator.py
------------------------
Generate molecular descriptors and fingerprints from SMILES using RDKit.
Supports:
  - RDKit physicochemical descriptors (~200 features)
  - Morgan circular fingerprints (bit vector)
  - MACCS keys (167-bit)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors


# All available RDKit descriptor names
_DESCRIPTOR_NAMES: List[str] = [name for name, _ in Descriptors.descList]


def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit Mol object. Returns None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None


def generate_rdkit_descriptors(
    smiles_list: List[str],
    descriptor_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Generate RDKit physicochemical descriptors.

    Parameters
    ----------
    smiles_list     : list of SMILES strings
    descriptor_names: optional subset of descriptor names; uses all if None

    Returns
    -------
    desc_df   : DataFrame of shape (n_valid, n_descriptors)
    failed_idx: list of indices where mol conversion failed
    """
    names = descriptor_names if descriptor_names else _DESCRIPTOR_NAMES
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)

    rows = []
    failed_idx = []

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            failed_idx.append(i)
            rows.append([np.nan] * len(names))
        else:
            try:
                desc = calc.CalcDescriptors(mol)
                rows.append(list(desc))
            except Exception:
                failed_idx.append(i)
                rows.append([np.nan] * len(names))

    df = pd.DataFrame(rows, columns=names)
    # Replace true inf with NaN, then clip everything to float32 limits
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_f32 = float(np.finfo(np.float32).max)
    df = df.clip(lower=-max_f32, upper=max_f32)
    return df, failed_idx


def generate_morgan_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Generate Morgan (ECFP) circular fingerprints as bit vectors.

    Parameters
    ----------
    smiles_list : list of SMILES strings
    radius      : Morgan radius (2 = ECFP4, 3 = ECFP6)
    n_bits      : fingerprint bit length

    Returns
    -------
    fp_df     : DataFrame of shape (n, n_bits)
    failed_idx: list of indices where conversion failed
    """
    col_names = [f"Morgan_r{radius}_{i}" for i in range(n_bits)]
    rows = []
    failed_idx = []

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            failed_idx.append(i)
            rows.append([0] * n_bits)
        else:
            try:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                rows.append(list(fp))
            except Exception:
                failed_idx.append(i)
                rows.append([0] * n_bits)

    return pd.DataFrame(rows, columns=col_names), failed_idx


def generate_maccs_keys(
    smiles_list: List[str],
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Generate 167-bit MACCS structural keys.

    Returns
    -------
    maccs_df  : DataFrame of shape (n, 167)
    failed_idx: list of indices where conversion failed
    """
    col_names = [f"MACCS_{i}" for i in range(167)]
    rows = []
    failed_idx = []

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            failed_idx.append(i)
            rows.append([0] * 167)
        else:
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                rows.append(list(fp))
            except Exception:
                failed_idx.append(i)
                rows.append([0] * 167)

    return pd.DataFrame(rows, columns=col_names), failed_idx


def generate_all_features(
    smiles_list: List[str],
    use_rdkit: bool = True,
    use_morgan: bool = False,
    use_maccs: bool = False,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Combine selected descriptor types into a single feature DataFrame.

    Returns
    -------
    full_df   : concatenated feature DataFrame
    failed_idx: union of all failed indices
    """
    dfs = []
    all_failed = set()

    if use_rdkit:
        df_rd, fi = generate_rdkit_descriptors(smiles_list)
        dfs.append(df_rd)
        all_failed.update(fi)

    if use_morgan:
        df_mo, fi = generate_morgan_fingerprints(smiles_list, radius=morgan_radius, n_bits=morgan_bits)
        dfs.append(df_mo)
        all_failed.update(fi)

    if use_maccs:
        df_ma, fi = generate_maccs_keys(smiles_list)
        dfs.append(df_ma)
        all_failed.update(fi)

    if not dfs:
        return pd.DataFrame(), []

    full_df = pd.concat(dfs, axis=1)
    return full_df, sorted(all_failed)
