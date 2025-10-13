import os, re, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# -------- Paths via env (no hard-coded secrets) --------
DATA_CORES_UBC = os.getenv("BK_DATA_CORES_UBC_DIR", "").rstrip("/")
DATA_CORES_QUEENS = os.getenv("BK_DATA_CORES_QUEENS_DIR", "").rstrip("/")
RF_SIGNAL_DIR = os.getenv("BK_RF_SIGNAL_DIR", "").rstrip("/")
OTHERCORES_INFO_DIR = os.getenv("BK_OTHERCORES_INFO_DIR", "").rstrip("/")

def _require_dir(path: str, var_name: str):
    if not path:
        raise RuntimeError(f"Set {var_name} to your data directory.")
    if not os.path.isdir(path):
        raise RuntimeError(f"{var_name} directory not found: {path}")
    return path

def _safe_float(x, default=0.0):
    try: return float(x)
    except Exception: return float(default)

def extract_info_json(info_array, df, _pandas_idx=0):
    dicts = [df] if df.shape[0] > 0 else []
    for info in info_array:
        with open(info, "r") as f:
            data = json.load(f)
        center = re.findall(r"BK_(\w+)_CORES", info)[0]
        core_idx = re.findall(r"pat(\d+)_cor(\d+)", info)[0]
        core_id = f"{int(core_idx[0]):04d}.{core_idx[1]}"
        entry = {
            "core_id": f"{center.lower()}_{core_id}",
            "center": center,
            "patient_id": f"{center.lower()}_{int(core_idx[0]):04d}",
            "inv": _safe_float(data.get("Involvement", 0.0)) / 100.0,
            "pathology": data.get("Pathology", ""),
            "label": 1 if data.get("Pathology", "") == "Adenocarcinoma" else 0,
            "filetemplate": info.replace("_info.json", ""),
            "Prostate_size": data.get("Prostate size", np.nan),
            "PSA": data.get("PSA", np.nan),
            "CoreName": data.get("CoreName", ""),
        }
        dicts.append(pd.DataFrame(entry, index=[_pandas_idx]))
        _pandas_idx += 1
    return pd.concat(dicts) if dicts else df

def make_table(ubc=True, queens=True):
    df = pd.DataFrame(columns=[
        "core_id","patient_id","inv","pathology","label","Prostate_size","PSA","CoreName","filetemplate","center"
    ])
    if ubc:
        ubc_path = _require_dir(DATA_CORES_UBC, "BK_DATA_CORES_UBC_DIR")
        ubc_info = [os.path.join(ubc_path, f) for f in os.listdir(ubc_path) if f.endswith(".json")]
        df = extract_info_json(ubc_info, df)
        df.inv = df.inv.apply(lambda x: x * 100.0)  # keep legacy fix
    if queens:
        queens_path = _require_dir(DATA_CORES_QUEENS, "BK_DATA_CORES_QUEENS_DIR")
        queens_info = [os.path.join(queens_path, f) for f in os.listdir(queens_path) if f.endswith(".json")]
        df = extract_info_json(queens_info, df, df.shape[0] - 1)
    return df

def select_patients(patient_ids):
    table = make_table(ubc=True, queens=True)
    return table[table.patient_id.isin(patient_ids)]

def split_patients(seed=0, oversample_cancer=False, undersample_benign=False, sampling_ratio=1):
    table = make_table(ubc=True, queens=True)
    patient_table = table.drop_duplicates(subset=["patient_id"])
    train_pa, val_pa = train_test_split(patient_table, test_size=0.3, random_state=seed, stratify=patient_table["label"])
    val_pa, test_pa = train_test_split(val_pa, test_size=0.5, random_state=seed, stratify=val_pa["label"])
    train_idx = table.patient_id.isin(train_pa.patient_id)
    val_idx   = table.patient_id.isin(val_pa.patient_id)
    test_idx  = table.patient_id.isin(test_pa.patient_id)
    train_tab, val_tab, test_tab = table[train_idx], table[val_idx], table[test_idx]

    num_benign = (train_tab.label == 0).sum()
    num_cancer = (train_tab.label == 1).sum()
    if oversample_cancer and num_cancer > 0:
        num_resample = max(int(num_benign * sampling_ratio) - num_cancer, 0)
        if num_resample > 0:
            train_tab = pd.concat([train_tab, train_tab[train_tab.label == 1].sample(num_resample, replace=True, random_state=seed)]).reset_index(drop=True)
    elif undersample_benign and num_cancer > 0:
        num_resample = int(num_cancer / max(sampling_ratio, 1))
        train_tab = pd.concat([train_tab[train_tab.label == 1], train_tab[train_tab.label == 0].sample(num_resample, random_state=seed)]).reset_index(drop=True)

    # sanity: disjoint patients
    assert set(train_tab.patient_id) & set(val_tab.patient_id) == set()
    assert set(train_tab.patient_id) & set(test_tab.patient_id) == set()
    assert set(val_tab.patient_id) & set(test_tab.patient_id) == set()
    return train_tab, val_tab, test_tab

def split_centerwise(inv_threshold=None):
    table = make_table(ubc=True, queens=True)
    if inv_threshold:
        table = table[(table["inv"] >= inv_threshold) | (table["label"] == 0)]
    ubc_patients = table[table.center == "UBC"].drop_duplicates(subset=["patient_id"])
    queens_table = table[table.center == "QUEENS"]

    val_pa, test_pa = train_test_split(ubc_patients, test_size=0.5, random_state=0, stratify=ubc_patients["label"])
    train_idx = table.patient_id.isin(queens_table.patient_id)
    val_idx   = table.patient_id.isin(val_pa.patient_id)
    test_idx  = table.patient_id.isin(test_pa.patient_id)
    train_tab, val_tab, test_tab = table[train_idx], table[val_idx], table[test_idx]

    assert set(train_tab.patient_id) & set(val_tab.patient_id) == set()
    assert set(train_tab.patient_id) & set(test_tab.patient_id) == set()
    assert set(val_tab.patient_id) & set(test_tab.patient_id) == set()
    return train_tab, val_tab, test_tab

def split_patients_kfold(fold_id, k=5, seed=0, centers=("UBC","QUEENS"),
                         oversample_cancer=False, undersample_benign=False, sampling_ratio=1):
    table = make_table(ubc=("UBC" in centers), queens=("QUEENS" in centers))
    patient_table = table.drop_duplicates(subset=["patient_id"])
    skf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    for i, (tr_idx, te_idx) in enumerate(skf.split(patient_table, patient_table["label"])):
        if i != fold_id: continue
        train_pa = patient_table.iloc[tr_idx]
        tmp = patient_table.iloc[te_idx]
        val_pa, test_pa = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=tmp["label"])
        train_idx = table.patient_id.isin(train_pa.patient_id)
        val_idx   = table.patient_id.isin(val_pa.patient_id)
        test_idx  = table.patient_id.isin(test_pa.patient_id)
        train_tab, val_tab, test_tab = table[train_idx], table[val_idx], table[test_idx]

        num_benign = (train_tab.label == 0).sum()
        num_cancer = (train_tab.label == 1).sum()
        if oversample_cancer and num_cancer > 0:
            num_resample = max(int(num_benign * sampling_ratio) - num_cancer, 0)
            if num_resample > 0:
                train_tab = pd.concat([train_tab, train_tab[train_tab.label == 1].sample(num_resample, replace=True, random_state=seed)]).reset_index(drop=True)
        elif undersample_benign and num_cancer > 0:
            num_resample = int(num_cancer / max(sampling_ratio, 1))
            train_tab = pd.concat([train_tab[train_tab.label == 1], train_tab[train_tab.label == 0].sample(num_resample, random_state=seed)]).reset_index(drop=True)

        return train_tab, val_tab, test_tab

    raise ValueError(f"fold_id {fold_id} out of range for k={k}")

def aux_paths_for_filetemplate(extracted_basename):
    signal_file = os.path.join(RF_SIGNAL_DIR, f"{extracted_basename}_signal.npy") if RF_SIGNAL_DIR else ""
    other_cores_path = os.path.join(OTHERCORES_INFO_DIR, f"{extracted_basename}.json") if OTHERCORES_INFO_DIR else ""
    other_cores_info = ""
    if other_cores_path and os.path.exists(other_cores_path):
        try:
            with open(other_cores_path, "r") as fh:
                data = json.load(fh)
            v = data.get("OtherCoresInfo", "")
            other_cores_info = "" if (isinstance(v, float) and np.isnan(v)) else str(v)
        except Exception:
            other_cores_info = ""
    return signal_file, other_cores_info
