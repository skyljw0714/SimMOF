#!/usr/bin/env python

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MOFSIMPLIFY_ROOT = Path(__file__).resolve().parent.parent / "working_dir/.venvs/mofsimplify_models/MOFSimplify"
ZEO_BIN = str(Path(__file__).resolve().parent.parent / "Zeopp/network")

SOLVENT_MODEL_PATH = str(MOFSIMPLIFY_ROOT / "model/solvent/ANN/final_model_flag_few_epochs.h5")
THERMAL_MODEL_PATH = str(MOFSIMPLIFY_ROOT / "model/thermal/ANN/final_model_T_few_epochs.h5")
SOLVENT_TRAIN_CSV = str(MOFSIMPLIFY_ROOT / "model/solvent/ANN/dropped_connectivity_dupes/train.csv")
THERMAL_TRAIN_CSV = str(MOFSIMPLIFY_ROOT / "model/thermal/ANN/train.csv")

RAC_FEATURES = [
    'D_func-I-0-all','D_func-I-1-all','D_func-I-2-all','D_func-I-3-all',
    'D_func-S-0-all','D_func-S-1-all','D_func-S-2-all','D_func-S-3-all',
    'D_func-T-0-all','D_func-T-1-all','D_func-T-2-all','D_func-T-3-all',
    'D_func-Z-0-all','D_func-Z-1-all','D_func-Z-2-all','D_func-Z-3-all',
    'D_func-chi-0-all','D_func-chi-1-all','D_func-chi-2-all','D_func-chi-3-all',
    'D_lc-I-0-all','D_lc-I-1-all','D_lc-I-2-all','D_lc-I-3-all',
    'D_lc-S-0-all','D_lc-S-1-all','D_lc-S-2-all','D_lc-S-3-all',
    'D_lc-T-0-all','D_lc-T-1-all','D_lc-T-2-all','D_lc-T-3-all',
    'D_lc-Z-0-all','D_lc-Z-1-all','D_lc-Z-2-all','D_lc-Z-3-all',
    'D_lc-chi-0-all','D_lc-chi-1-all','D_lc-chi-2-all','D_lc-chi-3-all',
    'D_mc-I-0-all','D_mc-I-1-all','D_mc-I-2-all','D_mc-I-3-all',
    'D_mc-S-0-all','D_mc-S-1-all','D_mc-S-2-all','D_mc-S-3-all',
    'D_mc-T-0-all','D_mc-T-1-all','D_mc-T-2-all','D_mc-T-3-all',
    'D_mc-Z-0-all','D_mc-Z-1-all','D_mc-Z-2-all','D_mc-Z-3-all',
    'D_mc-chi-0-all','D_mc-chi-1-all','D_mc-chi-2-all','D_mc-chi-3-all',
    'f-I-0-all','f-I-1-all','f-I-2-all','f-I-3-all',
    'f-S-0-all','f-S-1-all','f-S-2-all','f-S-3-all',
    'f-T-0-all','f-T-1-all','f-T-2-all','f-T-3-all',
    'f-Z-0-all','f-Z-1-all','f-Z-2-all','f-Z-3-all',
    'f-chi-0-all','f-chi-1-all','f-chi-2-all','f-chi-3-all',
    'f-lig-I-0','f-lig-I-1','f-lig-I-2','f-lig-I-3',
    'f-lig-S-0','f-lig-S-1','f-lig-S-2','f-lig-S-3',
    'f-lig-T-0','f-lig-T-1','f-lig-T-2','f-lig-T-3',
    'f-lig-Z-0','f-lig-Z-1','f-lig-Z-2','f-lig-Z-3',
    'f-lig-chi-0','f-lig-chi-1','f-lig-chi-2','f-lig-chi-3',
    'func-I-0-all','func-I-1-all','func-I-2-all','func-I-3-all',
    'func-S-0-all','func-S-1-all','func-S-2-all','func-S-3-all',
    'func-T-0-all','func-T-1-all','func-T-2-all','func-T-3-all',
    'func-Z-0-all','func-Z-1-all','func-Z-2-all','func-Z-3-all',
    'func-chi-0-all','func-chi-1-all','func-chi-2-all','func-chi-3-all',
    'lc-I-0-all','lc-I-1-all','lc-I-2-all','lc-I-3-all',
    'lc-S-0-all','lc-S-1-all','lc-S-2-all','lc-S-3-all',
    'lc-T-0-all','lc-T-1-all','lc-T-2-all','lc-T-3-all',
    'lc-Z-0-all','lc-Z-1-all','lc-Z-2-all','lc-Z-3-all',
    'lc-chi-0-all','lc-chi-1-all','lc-chi-2-all','lc-chi-3-all',
    'mc-I-0-all','mc-I-1-all','mc-I-2-all','mc-I-3-all',
    'mc-S-0-all','mc-S-1-all','mc-S-2-all','mc-S-3-all',
    'mc-T-0-all','mc-T-1-all','mc-T-2-all','mc-T-3-all',
    'mc-Z-0-all','mc-Z-1-all','mc-Z-2-all','mc-Z-3-all',
    'mc-chi-0-all','mc-chi-1-all','mc-chi-2-all','mc-chi-3-all',
]
GEO_FEATURES = [
    'Df', 'Di', 'Dif', 'GPOAV', 'GPONAV', 'GPOV', 'GSA',
    'POAV', 'POAV_vol_frac', 'PONAV', 'PONAV_vol_frac', 'VPOV', 'VSA', 'cell_v',
]
ALL_FEATURES = RAC_FEATURES + GEO_FEATURES


def _run_zeopp(primitive_cif: str, name: str, work_dir: Path) -> Optional[dict]:
    pd_f = str(work_dir / f"{name}_pd.txt")
    sa_f = str(work_dir / f"{name}_sa.txt")
    pov_f = str(work_dir / f"{name}_pov.txt")
    cmds = [
        [ZEO_BIN, "-ha", "-res", pd_f, primitive_cif],
        [ZEO_BIN, "-sa", "1.86", "1.86", "10000", sa_f, primitive_cif],
        [ZEO_BIN, "-volpo", "1.86", "1.86", "10000", pov_f, primitive_cif],
    ]
    procs = [subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for c in cmds]
    for p in procs:
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            return None

    if not (os.path.exists(pd_f) and os.path.exists(sa_f) and os.path.exists(pov_f)):
        return None

    try:
        with open(pd_f) as f:
            row = f.readline().split()
            Di, Df, Dif = float(row[1]), float(row[2]), float(row[3])
        with open(sa_f) as f:
            line = f.readline()
            cell_v = float(line.split('Unitcell_volume:')[1].split()[0])
            VSA = float(line.split('ASA_m^2/cm^3:')[1].split()[0])
            GSA = float(line.split('ASA_m^2/g:')[1].split()[0])
        with open(pov_f) as f:
            line = f.readline()
            density = float(line.split('Density:')[1].split()[0])
            POAV = float(line.split('POAV_A^3:')[1].split()[0])
            PONAV = float(line.split('PONAV_A^3:')[1].split()[0])
            GPOAV = float(line.split('POAV_cm^3/g:')[1].split()[0])
            GPONAV = float(line.split('PONAV_cm^3/g:')[1].split()[0])
            POAV_vf = float(line.split('POAV_Volume_fraction:')[1].split()[0])
            PONAV_vf = float(line.split('PONAV_Volume_fraction:')[1].split()[0])
            VPOV = POAV_vf + PONAV_vf
            GPOV = VPOV / density if density != 0 else np.nan
    except Exception:
        return None

    return {
        'Di': Di, 'Df': Df, 'Dif': Dif, 'cell_v': cell_v,
        'VSA': VSA, 'GSA': GSA, 'VPOV': VPOV, 'GPOV': GPOV,
        'POAV': POAV, 'PONAV': PONAV, 'GPOAV': GPOAV, 'GPONAV': GPONAV,
        'POAV_vol_frac': POAV_vf, 'PONAV_vol_frac': PONAV_vf,
    }


def _get_rac_df(primitive_cif: str, name: str, rac_dir: Path) -> Optional[pd.DataFrame]:
    from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            full_names, full_descriptors = get_MOF_descriptors(
                primitive_cif, 3,
                path=str(rac_dir) + "/",
                xyzpath=str(rac_dir / f"{name}.xyz"),
            )
    except Exception:
        return None
    if len(full_names) <= 1:
        return None
    try:
        lc_df = pd.read_csv(rac_dir / "lc_descriptors.csv").mean(numeric_only=True).to_frame().T
        sbu_df = pd.read_csv(rac_dir / "sbu_descriptors.csv").mean(numeric_only=True).to_frame().T
        linker_df = pd.read_csv(rac_dir / "linker_descriptors.csv").mean(numeric_only=True).to_frame().T
    except Exception:
        return None
    return pd.concat([lc_df, sbu_df, linker_df], axis=1)


def _normalize_solvent(df_train: pd.DataFrame, df_new: pd.DataFrame, features: list):
    _train = df_train.copy().dropna(subset=features + ["flag"])
    X_tr = _train[features].values
    X_new = df_new[features].values
    sc = sklearn.preprocessing.StandardScaler().fit(X_tr)
    return sc.transform(X_new)


def _normalize_thermal(df_train: pd.DataFrame, df_new: pd.DataFrame, features: list):
    _train = df_train.copy().dropna(subset=features + ["T"])
    X_tr = _train[features].values
    y_tr = _train[["T"]].values
    sc_x = sklearn.preprocessing.StandardScaler().fit(X_tr)
    sc_y = sklearn.preprocessing.StandardScaler().fit(y_tr)
    return sc_x.transform(df_new[features].values), sc_y


def predict_one(cif_path: str, work_dir: Path, sol_model, therm_model,
                sol_train: pd.DataFrame, therm_train: pd.DataFrame) -> dict:
    name = Path(cif_path).stem
    mof_dir = work_dir / name
    mof_dir.mkdir(parents=True, exist_ok=True)

    from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive

    prim_cif = str(mof_dir / f"{name}_primitive.cif")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            get_primitive(cif_path, prim_cif)
    except Exception as e:
        return {"name": name, "error": f"get_primitive: {e}"}

    zeo_dir = mof_dir / "zeo"
    zeo_dir.mkdir(exist_ok=True)
    geo = _run_zeopp(prim_cif, name, zeo_dir)
    if geo is None:
        return {"name": name, "error": "zeo++ failed"}

    rac_dir = mof_dir / "RACs"
    rac_dir.mkdir(exist_ok=True)
    rac_df = _get_rac_df(prim_cif, name, rac_dir)
    if rac_df is None:
        return {"name": name, "error": "RAC featurization failed"}

    geo_df = pd.DataFrame([geo])
    merged = pd.concat([geo_df, rac_df], axis=1)

    result: dict = {"name": name}

    sol_train_var = sol_train.loc[:, (sol_train != sol_train.iloc[0]).any()]
    sol_features = [f for f in sol_train_var.columns if f in ALL_FEATURES and f in merged.columns]
    if sol_features:
        try:
            X = _normalize_solvent(sol_train_var, merged, sol_features)
            pred = float(sol_model.predict(X, verbose=0)[0][0])
            result["solvent_pred"] = round(pred, 3)
        except Exception as e:
            result["solvent_error"] = str(e)

    therm_train_var = therm_train.loc[:, (therm_train != therm_train.iloc[0]).any()]
    therm_features = [f for f in therm_train_var.columns if f in ALL_FEATURES and f in merged.columns]
    if therm_features:
        try:
            X, sc_y = _normalize_thermal(therm_train_var, merged, therm_features)
            pred_scaled = therm_model.predict(X, verbose=0)
            pred_C = float(sc_y.inverse_transform(pred_scaled)[0][0])
            result["thermal_pred_C"] = round(pred_C, 1)
        except Exception as e:
            result["thermal_error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--thermal-threshold", type=float, default=None)
    parser.add_argument("--solvent-threshold", type=float, default=None)
    args = parser.parse_args()

    thermal_thr = args.thermal_threshold
    solvent_thr = args.solvent_threshold

    cif_dir = Path(args.cif_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        print(json.dumps({"error": "no CIFs found", "results": []}))
        sys.exit(0)

    print(f"[MOFSimplify] Loading models...", file=sys.stderr)
    sol_model = tf.keras.models.load_model(SOLVENT_MODEL_PATH, compile=False)
    therm_model = tf.keras.models.load_model(THERMAL_MODEL_PATH, compile=False)
    sol_train = pd.read_csv(SOLVENT_TRAIN_CSV)
    therm_train = pd.read_csv(THERMAL_TRAIN_CSV)
    print(f"[MOFSimplify] Models loaded. Processing {len(cif_files)} CIFs...", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        results = []
        kept = []
        for cif in cif_files:
            r = predict_one(str(cif), work_dir, sol_model, therm_model, sol_train, therm_train)
            if "error" in r:
                print(f"[MOFSimplify] {cif.name}: FAILED — {r['error']}", file=sys.stderr)
                r["passes"] = False
            else:
                sol_ok = (solvent_thr is None) or (r.get("solvent_pred", 0.0) >= solvent_thr)
                therm_ok = (thermal_thr is None) or (r.get("thermal_pred_C", 0.0) >= thermal_thr)
                r["passes"] = sol_ok and therm_ok
                print(
                    f"[MOFSimplify] {cif.name}: solvent={r.get('solvent_pred','?')} "
                    f"thermal={r.get('thermal_pred_C','?')}°C passes={r['passes']}",
                    file=sys.stderr,
                )
                if r["passes"]:
                    dest = out_dir / cif.name
                    shutil.copy2(str(cif), str(dest))
                    kept.append(str(dest))
            results.append(r)

    output = {
        "thermal_threshold": thermal_thr,
        "solvent_threshold": solvent_thr,
        "total": len(cif_files),
        "kept": len(kept),
        "kept_paths": kept,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
