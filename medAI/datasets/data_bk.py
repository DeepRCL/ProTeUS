"""Biopsy core dataset for ProTeUS prostate cancer detection.

This module contains dataset classes for loading and processing biopsy core data,
including RF signals, B-mode images, and anatomical masks.
"""

import os, re, json
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

from .transforms import CorewiseTransform, RandomTranslation, AugmentThruTime
from .data_utils import (
    make_table, select_patients, split_patients, split_centerwise, split_patients_kfold,
    aux_paths_for_filetemplate
)

def _analytical_env(x):
    """Compute analytical envelope of RF signal.
    
    Args:
        x: Input RF signal array.
        
    Returns:
        Analytical envelope of the signal.
    """
    from scipy.signal import hilbert
    return np.abs(hilbert(x)) ** 0.3

class BKCorewiseDataset(Dataset):
    """Dataset class for biopsy core data.
    
    This dataset loads biopsy core data including RF signals, B-mode images,
    and anatomical masks for prostate cancer detection.
    
    Args:
        df: DataFrame containing core metadata.
        transform: Data transformation pipeline.
        im_sz: Image size for resizing.
        style: Frame selection style for B-mode generation.
    """
    
    def __init__(self, df, transform, im_sz=1024, style="avg_all"):
        super().__init__()
        self.data = self._collect_files(df)
        self.transform = transform
        self.table = df
        self.im_sz = im_sz
        self.style = style

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        rf_path, roi_path, wp_path, label, involvement, core_id, patient_id, *_extras = self.data[idx]
        roi_mask = np.load(roi_path)
        prostate_mask = np.load(wp_path)
        rf_file = np.load(rf_path)

        if self.style == "last_frame":
            bmode = _analytical_env(rf_file[:, :, -1])
        elif self.style == "avg_last_100":
            bmode = _analytical_env(rf_file[:, :, -100:].mean(axis=-1))
        elif self.style == "avg_all":
            bmode = _analytical_env(rf_file.mean(axis=-1))
        elif self.style == "random":
            frame_idx = np.random.randint(100, rf_file.shape[-1])
            bmode = _analytical_env(rf_file[:, :, frame_idx])
        elif self.style == "random_avg":
            frame_idx = np.random.randint(50, 150)
            bmode = _analytical_env(rf_file[:, :, frame_idx: frame_idx + 5].mean(axis=-1))
        else:
            bmode = _analytical_env(rf_file.mean(axis=-1))

        bmode = resize(bmode, (self.im_sz, self.im_sz))
        roi_mask = resize(roi_mask, (self.im_sz // 4, self.im_sz // 4))
        prostate_mask = resize(prostate_mask, (self.im_sz // 4, self.im_sz // 4))

        if self.transform is not None:
            bmode = torch.from_numpy(bmode).unsqueeze(0).float()
            roi_mask = torch.from_numpy(roi_mask).unsqueeze(0).float()
            prostate_mask = torch.from_numpy(prostate_mask).unsqueeze(0).float()
            bmode, roi_mask, prostate_mask = self.transform(bmode, roi_mask, prostate_mask)
            bmode = bmode.squeeze(0).numpy()
            roi_mask = roi_mask.squeeze(0).numpy()
            prostate_mask = prostate_mask.squeeze(0).numpy()

        return (bmode, roi_mask, prostate_mask, label, involvement, core_id, patient_id)

    def _collect_files(self, df):
        file_tuples = []
        for filetemplate in list(df.filetemplate):
            roi_file = f"{filetemplate}_needle.npy"
            wp_file = f"{filetemplate}_prostate.npy"
            rf_file = f"{filetemplate}_rf.npy"

            extracted = os.path.basename(filetemplate)
            signal_file, other_cores_info = aux_paths_for_filetemplate(extracted)

            core_suffix = extracted.replace("pat", "").replace("_cor", ".")
            patient_id_num = int(core_suffix.split(".")[0])
            core_idx = int(core_suffix.split(".")[1])
            center_pre = filetemplate.split("/")[3].split('_')[1].lower() if "/" in filetemplate else "ubc"
            core_id_key = f"{center_pre}_{patient_id_num:04d}.{core_idx}"

            sub_df = df[df.core_id == core_id_key]
            if sub_df.empty:
                continue
            label_values = int(sub_df.label.values[0])
            inv = float(sub_df.inv.values[0])
            core_id = sub_df.core_id.values[0]
            patient_id = sub_df.patient_id.values[0]
            prostate_size = sub_df.Prostate_size.values[0]
            psa = sub_df.PSA.values[0]
            core_name = sub_df.CoreName.values[0]

            file_tuples.append(
                (rf_file, roi_file, wp_file, label_values, inv, core_id, patient_id,
                 signal_file, other_cores_info, prostate_size, psa, core_name)
            )
        return file_tuples

class BKTemporalDataset(BKCorewiseDataset):
    def __getitem__(self, idx):
        (
            rf_path, roi_path, wp_path, label, involvement, core_id, patient_id,
            signal_file_path, othercores, prostate_size, psa, core_name
        ) = self.data[idx]

        roi_mask = np.load(roi_path)
        prostate_mask = np.load(wp_path)
        rf_file = np.load(rf_path)

        # optional aux data
        try:
            signal_file = np.load(signal_file_path) if signal_file_path else np.empty((0,))
        except Exception:
            signal_file = np.empty((0,))

        # ---------- choose ONE frame only ----------
        if self.style in {"first_frame", "first_and_last", "first_and_all"}:
            bmode = _analytical_env(rf_file[:, :, 0])
        elif self.style in {"last_frame", "last_and_all", "last_and_random"}:
            bmode = _analytical_env(rf_file[:, :, -1])
        elif self.style in {"random", "first_and_random"}:
            frame_idx = np.random.randint(0, rf_file.shape[-1])
            bmode = _analytical_env(rf_file[:, :, frame_idx])
        elif self.style == "avg":
            bmode = _analytical_env(np.mean(rf_file, axis=-1))
        else:
            # default: first frame
            bmode = _analytical_env(rf_file[:, :, 0])

        # ---------- resize ----------
        bmode = resize(bmode, (self.im_sz, self.im_sz))
        roi_mask = resize(roi_mask, (self.im_sz // 4, self.im_sz // 4))
        prostate_mask = resize(prostate_mask, (self.im_sz // 4, self.im_sz // 4))
        ood_mask = ((roi_mask + prostate_mask) < 0.5)

        # ---------- (optional) augmentation ----------
        if self.transform is not None:
            bmode = torch.from_numpy(bmode).unsqueeze(0).float()
            roi_mask = torch.from_numpy(roi_mask).unsqueeze(0).float()
            prostate_mask = torch.from_numpy(prostate_mask).unsqueeze(0).float()
            ood_mask = torch.from_numpy(ood_mask).unsqueeze(0).float()

            # use weak temporal aug on the single image, then a paired translation
            # (RandomTranslation already supports single or multiple tensors)
            bmode = self.transform("weak", bmode)
            bmode, roi_mask, prostate_mask, ood_mask = RandomTranslation()(
                bmode, roi_mask, prostate_mask, ood_mask
            )

            bmode = bmode.squeeze(0).numpy()
            roi_mask = roi_mask.squeeze(0).numpy()
            prostate_mask = prostate_mask.squeeze(0).numpy()
            ood_mask = ood_mask.squeeze(0).numpy()

        return (
            bmode,               # (H, W)
            roi_mask,            # (H/4, W/4)
            prostate_mask,       # (H/4, W/4)
            ood_mask,            # (H/4, W/4)
            label,               # int
            signal_file,         # np.array or empty
            othercores,          # str
            prostate_size,       # float/num
            psa,                 # float/num
            core_name,           # str
            involvement,         # float
            core_id,             # str
            patient_id           # str
        )

# ----------------- public dataloader helpers -----------------
def make_corewise_bk_dataloaders(
    batch_sz, im_sz=1024, style="avg_all", splitting="patients",
    fold=0, num_folds=5, seed=0, inv_threshold=None,
    centers=("UBC"), oversampling=False, undersampling=False, sampling_ratio=1
):
    if splitting == "patients":
        train_tab, val_tab, test_tab = split_patients(seed=seed, oversample_cancer=oversampling,
                                                      undersample_benign=undersampling, sampling_ratio=sampling_ratio)
    elif splitting == "centers":
        train_tab, val_tab, test_tab = split_centerwise()
    elif splitting == "from_file":
        from medAI.datasets.splits.bk_patient_splits import bk_patient_splits as splits
        train_tab = select_patients(splits["train"])
        val_tab   = select_patients(splits["val"])
        test_tab  = select_patients(splits["test"])
    elif splitting == "from_file_kfold":
        from medAI.datasets.splits.bk_patient_splits import bk_patient_splits as splits
        chosen = splits[fold]
        train_tab = select_patients(chosen["train"])
        val_tab   = select_patients(chosen["val"])
        test_tab  = select_patients(chosen["test"])
    elif splitting == "patients_kfold":
        train_tab, val_tab, test_tab = split_patients_kfold(
            fold, k=num_folds, seed=seed, centers=centers,
            oversample_cancer=oversampling, undersample_benign=undersampling, sampling_ratio=sampling_ratio
        )
    else:
        train_tab, val_tab, test_tab = split_patients(seed=seed)

    if inv_threshold:
        train_tab = train_tab[(train_tab.inv > float(inv_threshold)) | (train_tab.label == 0)]

    train_ds = BKCorewiseDataset(df=train_tab, transform=CorewiseTransform(), im_sz=im_sz, style=style)
    val_ds   = BKCorewiseDataset(df=val_tab, transform=None,                 im_sz=im_sz, style=style)
    test_ds  = BKCorewiseDataset(df=test_tab, transform=None,                im_sz=im_sz, style=style)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_sz, shuffle=False)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_sz, shuffle=False)
    return train_dl, val_dl, test_dl

def make_temporal_bk_dataloaders(
    batch_sz, im_sz=1024, style="first_frame", splitting="patients_kfold",
    fold=0, num_folds=5, seed=0, inv_threshold=None, centers=("UBC"),
    oversampling=False, undersampling=False, sampling_ratio=1, cores_to_exclude=None
):
    # same splitting options as above
    train_tab, val_tab, test_tab = None, None, None
    if splitting == "patients":
        train_tab, val_tab, test_tab = split_patients(seed=seed, oversample_cancer=oversampling,
                                                      undersample_benign=undersampling, sampling_ratio=sampling_ratio)
    elif splitting == "centers":
        train_tab, val_tab, test_tab = split_centerwise()
    elif splitting == "from_file":
        import json, pandas as pd
        with open("splits/bk_patient_splits.json", "r") as f:
            splits = json.load(f)
        train_tab = pd.read_csv(splits["train"])
        val_tab   = pd.read_csv(splits["val"])
        test_tab  = pd.read_csv(splits["test"])
    elif splitting == "from_file_kfold":
        from medAI.datasets.splits.bk_patient_splits import bk_patient_splits as splits
        chosen = splits[fold]
        train_tab = select_patients(chosen["train"])
        val_tab   = select_patients(chosen["val"])
        test_tab  = select_patients(chosen["test"])
    elif splitting == "patients_kfold":
        train_tab, val_tab, test_tab = split_patients_kfold(
            fold, k=num_folds, seed=seed, centers=centers,
            oversample_cancer=oversampling, undersample_benign=undersampling, sampling_ratio=sampling_ratio
        )
    else:
        train_tab, val_tab, test_tab = split_patients(seed=seed)

    if inv_threshold:
        train_tab = train_tab[(train_tab.inv > float(inv_threshold)) | (train_tab.label == 0)]

    # optional: restrict to UBC as in your original
    train_tab = train_tab[train_tab.center == "UBC"]
    val_tab   = val_tab[val_tab.center == "UBC"]
    test_tab  = test_tab[test_tab.center == "UBC"]

    if cores_to_exclude is not None and len(cores_to_exclude) > 0:
        train_tab = train_tab[~train_tab["core_id"].isin(cores_to_exclude)]

    train_ds = BKTemporalDataset(df=train_tab, transform=AugmentThruTime(), im_sz=im_sz, style=style)
    val_ds   = BKTemporalDataset(df=val_tab,   transform=None,               im_sz=im_sz, style="first_frame")
    test_ds  = BKTemporalDataset(df=test_tab,  transform=None,               im_sz=im_sz, style="first_frame")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_sz, shuffle=False)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_sz, shuffle=False)
    return train_dl, val_dl, test_dl
