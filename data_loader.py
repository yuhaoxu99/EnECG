import os
import wfdb
import ujson
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from statsmodels.tsa.seasonal import STL


def stl_resolve(data_raw, save_stl):
    if not os.path.exists(save_stl):
        os.makedirs(save_stl, exist_ok=True)

    trend_pk = os.path.join(save_stl, 'trend.pk')
    seasonal_pk = os.path.join(save_stl, 'seasonal.pk')
    resid_pk = os.path.join(save_stl, 'resid.pk')

    if all(os.path.isfile(x) for x in [trend_pk, seasonal_pk, resid_pk]):
        with open(trend_pk, 'rb') as f:
            trend_stamp = pickle.load(f)
        with open(seasonal_pk, 'rb') as f:
            seasonal_stamp = pickle.load(f)
        with open(resid_pk, 'rb') as f:
            resid_stamp = pickle.load(f)
    else:
        data_raw = np.array(data_raw)
        n, m = data_raw.shape  

        trend_stamp = torch.zeros([n, m], dtype=torch.float32)
        seasonal_stamp = torch.zeros([n, m], dtype=torch.float32)
        resid_stamp = torch.zeros([n, m], dtype=torch.float32)

        for i in range(m):
            res = STL(data_raw[:, i], period=50).fit()
            trend_stamp[:, i] = torch.tensor(res.trend, dtype=torch.float32)
            seasonal_stamp[:, i] = torch.tensor(res.seasonal, dtype=torch.float32)
            resid_stamp[:, i] = torch.tensor(res.resid, dtype=torch.float32)

        with open(trend_pk, 'wb') as f:
            pickle.dump(trend_stamp, f)
        with open(seasonal_pk, 'wb') as f:
            pickle.dump(seasonal_stamp, f)
        with open(resid_pk, 'wb') as f:
            pickle.dump(resid_stamp, f)

    return trend_stamp, seasonal_stamp, resid_stamp


def read_and_preprocess_ecg(ecg_path):
    try:
        signal_data = wfdb.rdsamp(ecg_path)[0]  # (length, channels)

        signal_data = transforms.ToTensor()(signal_data)  # (1, channels, length)
        signal_data = signal_data.squeeze(0).numpy()      # (channels, length)

        signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        if std_val != 0:
            signal_data = (signal_data - mean_val) / std_val
        else:
            signal_data = signal_data - mean_val

        return signal_data  # shape=(length, channels)
    except Exception as e:
        print(f"Error reading file at {ecg_path}: {e}")
        return None

def apply_downsample(signal_data, target_size):
    if target_size is not None:
        return resample(signal_data, target_size, axis=0)
    return signal_data


def load_data_all_modes(args):
    with open(args.file_path, 'r') as f:
        data = [ujson.loads(line) for line in f]
    df = pd.json_normalize(data)[["subject_id", "ecg_path", args.label]]

    features = []
    sub_ids_seen = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading all modes"):
        subj_id = int(row['subject_id'])
        if subj_id in sub_ids_seen:
            continue
        sub_ids_seen.add(subj_id)
        
        if len(sub_ids_seen) > 30000:
            break

        if args.task_name == "classification":
            label_val = int(row[args.label])
        else:
            label_val = float(row[args.label])
            if np.isclose(label_val, 29999.0):
                continue

        ecg_path = row["ecg_path"]
        raw_signal = read_and_preprocess_ecg(ecg_path)
        if raw_signal is None:
            continue

        fm_size = getattr(args, "downsample_size_fm", None)
        fm_data = apply_downsample(raw_signal, fm_size) 

        tempo_size = getattr(args, "downsample_size_tempo", None)
        tempo_data = apply_downsample(raw_signal, tempo_size)  
        feat_id_tempo = idx % tempo_data.shape[1]           
        tempo_data_single = tempo_data[:, feat_id_tempo:feat_id_tempo+1] 

        stl_path = f"/local/scratch/yxu81/PhysicialFM/stl/{ecg_path[-8:]}"
        trend_stamp, seasonal_stamp, resid_stamp = stl_resolve(tempo_data_single, stl_path)
        trend_stamp, seasonal_stamp, resid_stamp = (
            trend_stamp[:, feat_id_tempo:feat_id_tempo+1], 
            seasonal_stamp[:, feat_id_tempo:feat_id_tempo+1], 
            resid_stamp[:, feat_id_tempo:feat_id_tempo+1]
        )

        tsdl_size = getattr(args, "downsample_size_ts", None)
        tsdl_data = apply_downsample(raw_signal, tsdl_size)
        feat_id_tsdl = idx % tsdl_data.shape[1]
        tsdl_data_single = tsdl_data[:, feat_id_tsdl:feat_id_tsdl+1]

        feature_dict = {
            "subject_id": subj_id,
            "label": label_val,

            "fm_data": fm_data,
            "fm_downsample_size": fm_size,

            "tempo_data": tempo_data_single,
            "tempo_downsample_size": tempo_size,
            "trend_stamp": trend_stamp,
            "seasonal_stamp": seasonal_stamp,
            "resid_stamp": resid_stamp,

            "tsdl_data": tsdl_data_single,
            "tsdl_downsample_size": tsdl_size
        }
        features.append(feature_dict)

    features_df = pd.DataFrame(features)
    return features_df


class ECGDataset(Dataset):

    def __init__(self, features_df):
        self.features_df = features_df

    def __len__(self):
        return len(self.features_df)

    def __getitem__(self, idx):
        row = self.features_df.iloc[idx]
        fm_data = torch.tensor(row['signal_data'], dtype=torch.float32)

        if np.issubdtype(type(row['label']), np.integer):
            label = torch.tensor(row['label'], dtype=torch.long)
        else:
            label = torch.tensor(row['label'], dtype=torch.float32)

        return fm_data, label


class ECGDatasetTempo(Dataset):
    def __init__(self, features_df):
        self.features_df = features_df

    def __len__(self):
        return len(self.features_df)

    def __getitem__(self, idx):
        row = self.features_df.iloc[idx]

        tempo_data = torch.tensor(row['signal_data'], dtype=torch.float32)       # (length,1)
        trend_stamp = torch.tensor(row['trend_stamp'], dtype=torch.float32)     # (length,1)
        seasonal_stamp = torch.tensor(row['seasonal_stamp'], dtype=torch.float32)
        resid_stamp = torch.tensor(row['resid_stamp'], dtype=torch.float32)

        # label
        if np.issubdtype(type(row['label']), np.integer):
            label = torch.tensor(row['label'], dtype=torch.long)
        else:
            label = torch.tensor(row['label'], dtype=torch.float32)

        return tempo_data, label, trend_stamp, seasonal_stamp, resid_stamp
