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

##################################################
# 1) STL 分解函数（与原先保持一致）
##################################################
def stl_resolve(data_raw, save_stl):
    """对单通道信号进行 STL 分解并缓存."""
    if not os.path.exists(save_stl):
        os.makedirs(save_stl, exist_ok=True)

    trend_pk = os.path.join(save_stl, 'trend.pk')
    seasonal_pk = os.path.join(save_stl, 'seasonal.pk')
    resid_pk = os.path.join(save_stl, 'resid.pk')

    # 如果都有缓存文件，则直接读取
    if all(os.path.isfile(x) for x in [trend_pk, seasonal_pk, resid_pk]):
        with open(trend_pk, 'rb') as f:
            trend_stamp = pickle.load(f)
        with open(seasonal_pk, 'rb') as f:
            seasonal_stamp = pickle.load(f)
        with open(resid_pk, 'rb') as f:
            resid_stamp = pickle.load(f)
    else:
        data_raw = np.array(data_raw)  # shape: (length, 1)
        n, m = data_raw.shape  # m 通常是 1

        trend_stamp = torch.zeros([n, m], dtype=torch.float32)
        seasonal_stamp = torch.zeros([n, m], dtype=torch.float32)
        resid_stamp = torch.zeros([n, m], dtype=torch.float32)

        for i in range(m):
            # period=50 可根据实际需要进行调整
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

##################################################
# 2) 基础的 ECG 读取 & 预处理
##################################################
def read_and_preprocess_ecg(ecg_path):
    """
    读取 WFDB 的 ECG 文件，做基础预处理：ToTensor -> squeeze -> 去除 NaN -> Z-score。
    返回一个 ndarray: shape=(length, channels).
    """
    try:
        # 从 .hea / .dat 等文件读取
        signal_data = wfdb.rdsamp(ecg_path)[0]  # (length, channels)

        # 用 torchvision.transforms.ToTensor() 简单转成 tensor，再转回 numpy
        # 这里是可选的，因为要做 squeeze 等处理
        signal_data = transforms.ToTensor()(signal_data)  # (1, channels, length)
        signal_data = signal_data.squeeze(0).numpy()      # (channels, length)

        # 转置到 (length, channels)，与后面 resample、STL 习惯对齐
        #signal_data = signal_data.T  # (length, channels)

        # 去除 NaN / inf
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
    """
    如果 target_size 不为 None，则对 signal_data 在 axis=0 维度做重采样。
    否则，直接原样返回。
    """
    if target_size is not None:
        return resample(signal_data, target_size, axis=0)
    return signal_data

##################################################
# 3) 主函数：同时输出 FM/Tempo/TSDL 三套数据
##################################################
def load_data_all_modes(args):
    """
    一次性为每条记录产出三种模式的数据:
      - FM: 可能下采样，多通道
      - Tempo: 下采样后，抽取单通道 + STL 分解
      - TSDL: 下采样后，抽取单通道 (无 STL)
    返回的 DataFrame 每行包含:
      subject_id,
      label,
      fm_data, fm_downsample_size,
      tempo_data, tempo_downsample_size, trend_stamp, seasonal_stamp, resid_stamp,
      tsdl_data, tsdl_downsample_size
    """
    # 1) 读取 JSON 文件
    with open(args.file_path, 'r') as f:
        data = [ujson.loads(line) for line in f]
    df = pd.json_normalize(data)[["subject_id", "ecg_path", args.label]]

    features = []
    sub_ids_seen = set()

    # 2) 遍历每条记录
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading all modes"):
        subj_id = int(row['subject_id'])
        if subj_id in sub_ids_seen:
            continue
        sub_ids_seen.add(subj_id)

        # 如果只想最多 300 个 subject:
        if len(sub_ids_seen) > 30000:
            break

        # 解析 label
        if args.task_name == "classification":
            label_val = int(row[args.label])
        else:
            label_val = float(row[args.label])
            # 过滤特定无效值
            if np.isclose(label_val, 29999.0):
                continue

        ecg_path = row["ecg_path"]
        # 3) 读取+预处理 ECG 原始多通道
        raw_signal = read_and_preprocess_ecg(ecg_path)
        if raw_signal is None:
            continue

        # 4) 各模式下采样 + 单通道 / STL
        # 4.1) FM: 多通道
        fm_size = getattr(args, "downsample_size_fm", None)
        fm_data = apply_downsample(raw_signal, fm_size)  # 多通道

        # 4.2) Tempo: 单通道 + STL
        tempo_size = getattr(args, "downsample_size_tempo", None)
        tempo_data = apply_downsample(raw_signal, tempo_size)  # 还是多通道
        feat_id_tempo = idx % tempo_data.shape[1]              # 抽取单通道
        tempo_data_single = tempo_data[:, feat_id_tempo:feat_id_tempo+1]  # shape=(length,1)

        # 做 STL
        stl_path = f"/local/scratch/yxu81/PhysicialFM/stl/{ecg_path[-8:]}"
        trend_stamp, seasonal_stamp, resid_stamp = stl_resolve(tempo_data_single, stl_path)
        trend_stamp, seasonal_stamp, resid_stamp = (
            trend_stamp[:, feat_id_tempo:feat_id_tempo+1], 
            seasonal_stamp[:, feat_id_tempo:feat_id_tempo+1], 
            resid_stamp[:, feat_id_tempo:feat_id_tempo+1]
        )

        # 4.3) TSDL: 单通道 (无 STL)
        tsdl_size = getattr(args, "downsample_size_ts", None)
        tsdl_data = apply_downsample(raw_signal, tsdl_size)
        feat_id_tsdl = idx % tsdl_data.shape[1]
        tsdl_data_single = tsdl_data[:, feat_id_tsdl:feat_id_tsdl+1]

        # 5) 汇总保存进 feature_dict
        feature_dict = {
            "subject_id": subj_id,
            "label": label_val,

            # FM
            "fm_data": fm_data,
            "fm_downsample_size": fm_size,

            # Tempo
            "tempo_data": tempo_data_single,
            "tempo_downsample_size": tempo_size,
            "trend_stamp": trend_stamp,
            "seasonal_stamp": seasonal_stamp,
            "resid_stamp": resid_stamp,

            # TSDL
            "tsdl_data": tsdl_data_single,
            "tsdl_downsample_size": tsdl_size
        }
        features.append(feature_dict)

    # 转成 DataFrame 返回
    features_df = pd.DataFrame(features)
    return features_df

##################################################
# 4) 两个 Dataset 示例
##################################################
class ECGDataset(Dataset):
    """
    假设只需要用 FM 数据来做训练/推理，
    那么在 __getitem__ 里取 features_df['fm_data'] 就可以。
    """
    def __init__(self, features_df):
        self.features_df = features_df

    def __len__(self):
        return len(self.features_df)

    def __getitem__(self, idx):
        row = self.features_df.iloc[idx]
        fm_data = torch.tensor(row['signal_data'], dtype=torch.float32)  # shape=(length, channels or 1)

        # label
        if np.issubdtype(type(row['label']), np.integer):
            label = torch.tensor(row['label'], dtype=torch.long)
        else:
            label = torch.tensor(row['label'], dtype=torch.float32)

        return fm_data, label


class ECGDatasetTempo(Dataset):
    """
    若只想用 Tempo 模式的数据 (信号+STL)，
    那么这里在 __getitem__ 提取相应字段即可。
    """
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