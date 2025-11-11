#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_sif_sgf_features.py
---------------------------
Read a CSV with columns: id, SMILES, SIF_class, SGF_class
and compute peptide features for SIF/SGF stability modeling.

Output: a NumPy .npz file containing arrays:
  - X: feature matrix (n_samples, n_features)
  - y_sif: integer class labels for SIF stability
  - y_sgf: integer class labels for SGF stability
  - ids: original IDs (object dtype)
  - feature_names: list of feature names (object array of strings)
  - mask_valid: bool mask of successfully featurized rows (useful if input had invalid SMILES)

Notes:
  * Features follow guidance from your PPT: Morgan/Avalon fingerprints + QED properties
    and physchem descriptors capturing lipophilicity, rigidity, and size.
  * Avalon fingerprint is optional (depends on your RDKit build). If unavailable, it's skipped.
  * Requires: pandas, numpy, rdkit
    - pip install pandas numpy rdkit-pypi
"""
import argparse
import sys
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, QED, rdMolDescriptors, Crippen, Descriptors

# Try import Avalon fingerprint support (optional)
try:
    from rdkit.Avalon import pyAvalonTools
    _HAS_AVALON = True
except Exception:
    pyAvalonTools = None
    _HAS_AVALON = False


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def gasteiger_charge_stats(mol):
    """Compute Gasteiger charges and return summary stats (mean, max, min, std, sum)."""
    try:
        AllChem.ComputeGasteigerCharges(mol)
        vals = []
        for i in range(mol.GetNumAtoms()):
            try:
                vals.append(float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')))
            except Exception:
                vals.append(0.0)
        if len(vals) == 0:
            vals = [0.0]
        arr = np.asarray(vals, dtype=float)
        # Replace inf/nan with 0
        arr = np.where(np.isfinite(arr), arr, 0.0)
        mean = float(np.mean(arr))
        vmax = float(np.max(arr))
        vmin = float(np.min(arr))
        std = float(np.std(arr))
        s = float(np.sum(arr))
        return [mean, vmax, vmin, std, s]
    except Exception:
        # In case charge computation fails
        return [0.0, 0.0, 0.0, 0.0, 0.0]


def qed_props(mol):
    """QED properties vector: MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS"""
    try:
        p = QED.properties(mol)
        return [float(p.MW), float(p.ALOGP), float(p.HBA), float(p.HBD),
                float(p.PSA), float(p.ROTB), float(p.AROM), float(p.ALERTS)]
    except Exception:
        return [0.0]*8


def physchem_block(mol):
    """Physchem descriptors related to lipophilicity, rigidity, and size."""
    try:
        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        hba = float(rdMolDescriptors.CalcNumHBA(mol))
        hbd = float(rdMolDescriptors.CalcNumHBD(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        rotb = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        ring = float(rdMolDescriptors.CalcNumRings(mol))
        fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
        heavy = float(Descriptors.HeavyAtomCount(mol))
        n_atoms = float(mol.GetNumAtoms())
        # A simple rigidity proxy: rings / (1 + rotatable bonds)
        rigidity_proxy = ring / (1.0 + rotb)
        return [mw, logp, hba, hbd, tpsa, rotb, ring, fsp3, heavy, n_atoms, rigidity_proxy]
    except Exception:
        return [0.0]*11


def morgan_bits(mol, radius=2, n_bits=1024, use_chirality=True):
    try:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=int(n_bits), useChirality=use_chirality)
        arr = np.zeros((1, int(n_bits)), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(bv, arr[0])
        return arr[0].astype(np.float32).tolist()
    except Exception:
        return [0.0]*int(n_bits)


def avalon_bits(mol, n_bits=512):
    if not _HAS_AVALON:
        return None  # signify absence
    try:
        bv = pyAvalonTools.GetAvalonFP(mol, int(n_bits))
        arr = np.zeros((1, int(n_bits)), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(bv, arr[0])
        return arr[0].astype(np.float32).tolist()
    except Exception:
        return [0.0]*int(n_bits)


def build_feature_names(n_morgan=1024, n_avalon=512, have_avalon=True):
    names = []
    # QED props
    names += [f"QED_{k}" for k in ["MW","ALOGP","HBA","HBD","PSA","ROTB","AROM","ALERTS"]]
    # Physchem
    names += ["PC_MolWt","PC_LogP","PC_HBA","PC_HBD","PC_TPSA","PC_RotB","PC_Rings",
              "PC_FractionCSP3","PC_HeavyAtomCount","PC_NumAtoms","PC_RigidityProxy"]
    # Gasteiger charge stats
    names += ["GC_Mean","GC_Max","GC_Min","GC_Std","GC_Sum"]
    # Morgan bits
    names += [f"Morgan_{i}" for i in range(int(n_morgan))]
    # Avalon bits (optional)
    if have_avalon:
        names += [f"Avalon_{i}" for i in range(int(n_avalon))]
    return names


def featurize_smiles(smiles, n_morgan=1024, n_avalon=512, add_avalon=True):
    """Return (features_list, succeeded_bool, have_avalon) for a single SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, False, add_avalon and _HAS_AVALON

    # Base features
    f_qed = qed_props(mol)
    f_pc = physchem_block(mol)
    f_gc = gasteiger_charge_stats(mol)
    f_morgan = morgan_bits(mol, radius=2, n_bits=n_morgan, use_chirality=True)

    have_avalon = add_avalon and _HAS_AVALON
    if have_avalon:
        f_avalon = avalon_bits(mol, n_bits=n_avalon)
        feats = f_qed + f_pc + f_gc + f_morgan + f_avalon
    else:
        feats = f_qed + f_pc + f_gc + f_morgan

    return feats, True, have_avalon


def main():
    ap = argparse.ArgumentParser(description="Extract SIF/SGF peptide features to NumPy arrays.")
    ap.add_argument("--csv", required=True, help="Input CSV with header: id,SMILES,SIF_class,SGF_class")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--morgan_bits", type=int, default=1024, help="Number of Morgan bits (default: 1024)")
    ap.add_argument("--avalon_bits", type=int, default=512, help="Number of Avalon bits (default: 512)")
    ap.add_argument("--no_avalon", action="store_true", help="Disable Avalon fingerprint even if available")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = ["id", "SMILES", "SIF_class", "SGF_class"]
    for c in required_cols:
        if c not in df.columns:
            sys.stderr.write(f"[ERROR] Missing required column: {c}\n")
            sys.exit(2)

    X = []
    y_sif = []
    y_sgf = []
    ids = []
    mask_valid = []

    have_avalon_any = False

    for _, row in df.iterrows():
        _id = str(row["id"])
        smi = str(row["SMILES"])
        sif = row["SIF_class"]
        sgf = row["SGF_class"]

        feats, ok, have_avalon = featurize_smiles(
            smi,
            n_morgan=args.morgan_bits,
            n_avalon=args.avalon_bits,
            add_avalon=(not args.no_avalon),
        )
        have_avalon_any = have_avalon or have_avalon_any

        if ok and feats is not None:
            X.append(feats)
            y_sif.append(int(sif))
            y_sgf.append(int(sgf))
            ids.append(_id)
            mask_valid.append(True)
        else:
            # Keep placeholder row if you prefer, but here we skip invalid SMILES
            mask_valid.append(False)

    # Filter by mask_valid to align ids/labels with X
    if len(X) == 0:
        sys.stderr.write("[ERROR] No valid SMILES found.\n")
        sys.exit(3)

    ids = np.asarray([i for i, m in zip(df["id"].astype(str).tolist(), mask_valid) if m], dtype=object)
    y_sif = np.asarray([int(v) for v, m in zip(df["SIF_class"].tolist(), mask_valid) if m], dtype=np.int32)
    y_sgf = np.asarray([int(v) for v, m in zip(df["SGF_class"].tolist(), mask_valid) if m], dtype=np.int32)
    X = np.asarray(X, dtype=np.float32)
    mask_valid = np.asarray(mask_valid, dtype=bool)

    feature_names = build_feature_names(
        n_morgan=args.morgan_bits,
        n_avalon=args.avalon_bits,
        have_avalon=have_avalon_any
    )
    feature_names = np.asarray(feature_names, dtype=object)

    np.savez_compressed(
        args.out,
        X=X,
        y_sif=y_sif,
        y_sgf=y_sgf,
        ids=ids,
        feature_names=feature_names,
        mask_valid=mask_valid,
        meta=np.asarray({
            "morgan_bits": args.morgan_bits,
            "avalon_bits": args.avalon_bits,
            "avalon_included": have_avalon_any and (not args.no_avalon)
        }, dtype=object)
    )

    print(f"[OK] Saved features to: {args.out}")
    print(f" - Samples: {X.shape[0]}")
    print(f" - Features: {X.shape[1]}")
    print(f" - Avalon included: {have_avalon_any and (not args.no_avalon)}")


if __name__ == "__main__":
    main()
