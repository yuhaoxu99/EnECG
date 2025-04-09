#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import os
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_recall_fscore_support
)

from peft import LoraConfig, get_peft_model

from momentfm import MOMENTPipeline  # (1) MOMENT
from models.TEMPO import TEMPO       # (2) TEMPO
from fairseq_signals.models import build_model_from_checkpoint  # (3) ECG-FM
from models.DLinears import Model as DLinearModel               # (4) DLinear
from models.TimesNet import Model as TimesNetModel              # (5) TimesNet

from data_loader import (
    load_data_all_modes,
    ECGDataset,
    ECGDatasetTempo,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


def set_seed(seed):
    logger.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="MoE Ensemble of 5 models + FFN per expert + LoRA Demo")

    parser.add_argument("--file_path", type=str, default="/local/scratch/yxu81/PhysicialFM/instruct_data/mimic_ecg.jsonl")
    parser.add_argument("--label", type=str, default="age")
    parser.add_argument("--config_path", type=str, default="/local/scratch/yxu81/PhysicialFM/moment_config.yaml", help="MOMENT config")
    parser.add_argument("--task_name", type=str, default="regression", help="regression or classification")
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--downsample_size_fm", type=int, default=5000)
    parser.add_argument("--downsample_size_tempo", type=int, default=336)
    parser.add_argument("--downsample_size_ts", type=int, default=500)

    parser.add_argument("--ecgfm_ckpt", type=str, default="./ckpts/physionet_finetuned.pt")
    parser.add_argument("--dlinear_ckpt", type=str, default="./checkpoints/DLinear/checkpoint.pth")
    parser.add_argument("--timesnet_ckpt", type=str, default="./checkpoints/TimesNet/checkpoint.pth")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_lora", type=bool, default=True)

    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args


def load_model_moment(config):
    logger.info("Loading MOMENTPipeline (model1)")
    model = MOMENTPipeline.from_pretrained(
        config["model_path"],
        model_kwargs={
            "task_name": 'classification',
            "n_channels": config["n_channels"],
            "num_class": config['num_class'],
            "freeze_encoder": config['freeze_encoder'],
            "freeze_embedder": config['freeze_embedder']
        }
    )

    model.init()
    logger.info("MOMENTPipeline loaded.")
    return model


def load_model_tempo():
    logger.info("Loading TEMPO (model2)")
    model = TEMPO.load_pretrained_model(
        device=device,
        repo_id="Melady/TEMPO",
        filename="TEMPO-80M_v1.pth",
        cache_dir="./checkpoints/TEMPO_checkpoints"
    )
    model.eval()
    logger.info("TEMPO loaded.")
    return model


def load_model_ecgfm(ckpt_path):
    logger.info("Loading ECG-FM (model3)")
    model = build_model_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()
    logger.info("ECG-FM loaded.")
    return model


def load_model_dlinear(ckpt_path):
    logger.info("Loading DLinear (model4)")

    class Configs:
        seq_len = 500
        pred_len = 1
        label_len = 1
        d_model = 16
        d_ff = 32
        num_kernels = 6
        top_k = 3
        e_layers = 2
        enc_in = 12
        c_out = 1
        embed = 'fixed'
        freq = 'h'
        dropout = 0.3
        moving_avg = 1001
        weights = [1.0, 1.5]
        kernel_size = 1001
        task_name = "long_term_forecast"

    configs = Configs()
    model = DLinearModel(configs, 'cpu')
    ckp = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckp)
    model.eval()
    logger.info("DLinear loaded.")
    return model, configs


def load_model_timesnet(ckpt_path):
    logger.info("Loading TimesNet (model5)")

    class ConfigsTimes:
        seq_len = 500
        pred_len = 1
        label_len = 1
        d_model = 16
        d_ff = 32
        num_kernels = 6
        top_k = 3
        e_layers = 3
        enc_in = 1
        c_out = 1
        embed = 'timeF'
        freq = 'h'
        dropout = 0.3
        moving_avg = 251
        weights = [1.0, 1.5]
        kernel_size = 251
        task_name = "long_term_forecast"

    configs_t = ConfigsTimes()

    model = TimesNetModel(configs_t)
    ckp = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckp)
    model.eval()
    logger.info("TimesNet loaded.")
    return model, configs_t


class MoEEnsemble(nn.Module):

    def __init__(
        self,
        model_moment,
        model_tempo,
        model_ecgfm,
        model_dlinear,
        model_timesnet,
        task_name="regression",
        num_classes=1,
        freeze_experts=True
    ):
        super().__init__()
        self.task_name = task_name
        self.num_classes = num_classes

        self.model_moment = model_moment
        self.model_tempo = model_tempo
        self.model_ecgfm = model_ecgfm
        self.model_dlinear = model_dlinear
        self.model_timesnet = model_timesnet

        if freeze_experts:
            self._freeze_model_params(self.model_moment)
            self._freeze_model_params(self.model_tempo)
            self._freeze_model_params(self.model_ecgfm)
            self._freeze_model_params(self.model_dlinear)
            self._freeze_model_params(self.model_timesnet)

        out_dim = 1 if task_name == "regression" else num_classes

        def make_ffn(input_dim):
            hidden_dim=256
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )


        input_dim = out_dim

        self.ffn_moment = make_ffn(1)
        self.ffn_tempo = make_ffn(input_dim)
        self.ffn_ecgfm = make_ffn(input_dim)
        self.ffn_dlinear = make_ffn(1)
        self.ffn_timesnet = make_ffn(1)

        gating_input_dim = 5000
        gating_hidden_dim = 256
        self.gating_net = nn.Sequential(
            nn.Linear(gating_input_dim, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, 5)
        )

        self.projection_tempo = nn.Linear(96, out_dim).to(device)
        self.projection_ecgfm = nn.Linear(26, out_dim).to(device)

    def _freeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, 
                data_mom,         
                data_tempo_tuple, 
                data_dlin_tsnet
    ):

        out_mom = self.inference_moment(data_mom.permute(0,2,1))
        out_mom = self.ffn_moment(out_mom)

        (x_temp, trend_stamp, seasonal_stamp, resid_stamp) = data_tempo_tuple
        out_tempo = self.inference_tempo(x_temp, trend_stamp, seasonal_stamp, resid_stamp)
        out_tempo = self.ffn_tempo(out_tempo)

        out_ecgfm = self.inference_ecgfm(data_mom.permute(0,2,1))
        out_ecgfm = self.ffn_ecgfm(out_ecgfm)

        out_dlin = self.inference_dlinear(data_dlin_tsnet)
        out_dlin = self.ffn_dlinear(out_dlin)

        out_tn = self.inference_timesnet(data_dlin_tsnet)
        out_tn = self.ffn_timesnet(out_tn)

        gating_input = data_mom[:,:,1].to(device).float()
        # print("gating_input.shape:", gating_input.shape)
        gating_logits = self.gating_net(gating_input)
        gating_probs = F.softmax(gating_logits, dim=-1)


        experts_out = torch.stack([out_mom, out_tempo, out_ecgfm, out_dlin, out_tn], dim=1)

        gating_probs = gating_probs.unsqueeze(-1)
        weighted = experts_out * gating_probs
        output = weighted.sum(dim=1)

        return output


    def inference_moment(self, data):
        out = self.model_moment(data.to(device))
        logits = out.logits
        return logits

    def inference_tempo(self, x_temp, trend_stamp, seasonal_stamp, resid_stamp):

        x_temp = x_temp.to(device)
        trend_stamp = trend_stamp.to(device)
        seasonal_stamp = seasonal_stamp.to(device)
        resid_stamp = resid_stamp.to(device)

        enc_out, _, _ = self.model_tempo(x_temp, 0, trend_stamp, seasonal_stamp, resid_stamp)
        enc_out = enc_out.view(enc_out.size(0), -1)
        pred = self.projection_tempo(enc_out)
        return pred

    def inference_ecgfm(self, data):
        out_dict = self.model_ecgfm(source=data.float().to(device))
        x = out_dict["out"]
        x = self.projection_ecgfm(x)
        return x

    def inference_dlinear(self, data):
        x_in = data.to(device)
        out = self.model_dlinear(x_in, 0)
        if len(out.shape) == 3:
            out = out.squeeze(-1).squeeze(-1)
        if out.dim() == 1:
            out = out.unsqueeze(-1)

        return out

    def inference_timesnet(self, data):
        x_enc = data[:, :500, :].to(device)
        x_dec = data[:, -1:, :].to(device)
        out = self.model_timesnet(x_enc, None, x_dec, None)
        if len(out.shape) > 2:
            out = out.mean(dim=(1,2))
        out = out.unsqueeze(-1)
        return out


def train_one_epoch(model, train_loader, optimizer, criterion, task_name="regression"):
    model.train()
    total_loss = 0.0
    for batch_mom, batch_tempo, batch_dlin in train_loader:
        (data_mom, labels_m) = batch_mom
        (data_temp, labels_temp, trend_stamp, seasonal_stamp, resid_stamp) = batch_tempo
        (data_dl, labels_dl) = batch_dlin

        labels = labels_m.to(device).float()

        preds = model(
            data_mom, 
            (data_temp, trend_stamp, seasonal_stamp, resid_stamp),
            data_dl
        )

        if task_name == "classification":
            labels = labels.long()
            loss = criterion(preds, labels)
        else:
            preds = preds.view(-1)
            labels = labels.view(-1)
            loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_moe(model, test_loader, task_name="regression", num_classes=1):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_mom, batch_tempo, batch_dlin in test_loader:
            (data_mom, labels_m) = batch_mom
            (data_temp, labels_temp, trend_stamp, seasonal_stamp, resid_stamp) = batch_tempo
            (data_dl, labels_dl) = batch_dlin

            labels = labels_m.to(device).float()

            preds = model(
                data_mom,
                (data_temp, trend_stamp, seasonal_stamp, resid_stamp),
                data_dl
            )

            if task_name == "classification":
                pred_label = torch.argmax(preds, dim=-1).cpu().numpy()
                all_preds.extend(pred_label)
                all_labels.extend(labels_m.cpu().numpy())
            else:
                preds = preds.view(-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_m.view(-1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if task_name == "classification":
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}
    else:
        mae = mean_absolute_error(all_labels, all_preds)
        mse = mean_squared_error(all_labels, all_preds)
        return {"mae": mae, "mse": mse}


def main():
    set_seed(42)
    args = parse_args()

    logger.info("Loading data for all models (unified approach)")
    features_df_all = load_data_all_modes(args)
    logger.info("Data loaded successfully")

    features_df_moment = features_df_all.copy()
    features_df_moment.rename(columns={"fm_data": "signal_data"}, inplace=True)
    features_df_moment = features_df_moment[["subject_id", "label", "signal_data"]]

    features_df_tempo = features_df_all.copy()
    features_df_tempo.rename(columns={"tempo_data": "signal_data"}, inplace=True)
    features_df_tempo = features_df_tempo[[
        "subject_id", "label",
        "signal_data", "trend_stamp", "seasonal_stamp", "resid_stamp"
    ]]

    features_df_dlinear = features_df_all.copy()
    features_df_dlinear.rename(columns={"tsdl_data": "signal_data"}, inplace=True)
    features_df_dlinear = features_df_dlinear[["subject_id", "label", "signal_data"]]

    with open(args.config_path, "r") as f:
        moment_config = yaml.safe_load(f)

    model_moment = load_model_moment(moment_config)
    model_tempo = load_model_tempo()
    model_ecgfm = load_model_ecgfm(args.ecgfm_ckpt)
    model_dlinear, configs_dlin = load_model_dlinear(args.dlinear_ckpt)
    model_timesnet, configs_tn = load_model_timesnet(args.timesnet_ckpt)

    moe_model = MoEEnsemble(
        model_moment=model_moment,
        model_tempo=model_tempo,
        model_ecgfm=model_ecgfm,
        model_dlinear=model_dlinear,
        model_timesnet=model_timesnet,
        task_name=args.task_name,
        num_classes=args.num_class,
        freeze_experts=True
    ).to(device)

    if args.use_lora:
        try:
            target_modules = [
                "ffn_moment.0", "ffn_moment.2",
                "ffn_tempo.0",  "ffn_tempo.2",
                "ffn_ecgfm.0",  "ffn_ecgfm.2",
                "ffn_dlinear.0","ffn_dlinear.2",
                "ffn_timesnet.0","ffn_timesnet.2",
                "gating_net.0", "gating_net.2"
            ]
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=target_modules
            )
            moe_model = get_peft_model(moe_model, lora_config)
            logger.info("Enabled LoRA for 5 FFNs & gating network.")

    unique_subject_ids = features_df_moment["subject_id"].unique()
    num_splits = 3
    split_size = len(unique_subject_ids) // num_splits
    subject_id_splits = [
        unique_subject_ids[i*split_size : (i+1)*split_size] for i in range(num_splits)
    ]

    mae_list, mse_list = [], []
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    if args.task_name=="regression":
        criterion = nn.MSELoss()
    else: 
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(moe_model.parameters(), lr=args.lr)

    for fold_idx in range(num_splits):
        logger.info(f"===== Fold {fold_idx+1}/{num_splits} =====")
        test_ids = subject_id_splits[fold_idx]
        train_ids = np.concatenate(
            [subject_id_splits[i] for i in range(num_splits) if i!=fold_idx],
            axis=0
        )

        train_df_mom = features_df_moment[features_df_moment["subject_id"].isin(train_ids)]
        train_df_tempo = features_df_tempo[features_df_tempo["subject_id"].isin(train_ids)]
        train_df_dlin = features_df_dlinear[features_df_dlinear["subject_id"].isin(train_ids)]

        test_df_mom = features_df_moment[features_df_moment["subject_id"].isin(test_ids)]
        test_df_tempo = features_df_tempo[features_df_tempo["subject_id"].isin(test_ids)]
        test_df_dlin = features_df_dlinear[features_df_dlinear["subject_id"].isin(test_ids)]

        train_dataset_mom = ECGDataset(train_df_mom)
        train_dataset_tempo = ECGDatasetTempo(train_df_tempo)
        train_dataset_dlin = ECGDataset(train_df_dlin)

        test_dataset_mom = ECGDataset(test_df_mom)
        test_dataset_tempo = ECGDatasetTempo(test_df_tempo)
        test_dataset_dlin = ECGDataset(test_df_dlin)

        
        test_loader = zip(
            DataLoader(test_dataset_mom, batch_size=16, shuffle=False),
            DataLoader(test_dataset_tempo, batch_size=16, shuffle=False),
            DataLoader(test_dataset_dlin, batch_size=16, shuffle=False)
        )

        for epoch in range(args.epochs):
            moe_model.train()
            total_loss = 0.0
            total_steps = 0

            train_loader = zip(
                DataLoader(train_dataset_mom, batch_size=16, shuffle=True),
                DataLoader(train_dataset_tempo, batch_size=16, shuffle=True),
                DataLoader(train_dataset_dlin, batch_size=16, shuffle=True)
            )

            for (batch_mom, batch_tempo, batch_dlin) in train_loader:
                (data_mom, labels_m) = batch_mom
                (data_temp, labels_temp, trend_stamp, seasonal_stamp, resid_stamp) = batch_tempo
                (data_dl, labels_dl) = batch_dlin

                labels = labels_m.to(device).float()

                preds = moe_model(
                    data_mom,
                    (data_temp, trend_stamp, seasonal_stamp, resid_stamp),
                    data_dl
                )

                if args.task_name == "classification":
                    labels = labels.long()
                    loss = criterion(preds, labels)
                else:
                    preds = preds.view(-1)
                    labels = labels.view(-1)
                    loss = criterion(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1

            avg_loss = total_loss / max(total_steps, 1)
            logger.info(f"Fold{fold_idx+1} Epoch[{epoch+1}/{args.epochs}] - TrainLoss: {avg_loss:.4f}")

        metrics = evaluate_moe(moe_model, test_loader, args.task_name, args.num_class)
        if args.task_name == "classification":
            acc_list.append(metrics["acc"])
            prec_list.append(metrics["precision"])
            rec_list.append(metrics["recall"])
            f1_list.append(metrics["f1"])
            logger.info(f"Fold {fold_idx+1} - ACC: {metrics['acc']:.4f}, "
                        f"P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        else:
            mae_list.append(metrics["mae"])
            mse_list.append(metrics["mse"])
            logger.info(f"Fold {fold_idx+1} - MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}")

    if args.task_name == "classification":
        logger.info(
            f"3-Fold Results - "
            f"ACC: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}, "
            f"Precision: {np.mean(prec_list):.4f} ± {np.std(prec_list):.4f}, "
            f"Recall: {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}, "
            f"F1: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}"
        )
    else:
        logger.info(
            f"3-Fold Results - "
            f"MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}, "
            f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}"
        )

    if args.use_lora:
        logger.info("Saving LoRA adapter to: ./moe_lora_adapter/")
        moe_model.save_pretrained("./moe_lora_adapter")


if __name__ == "__main__":
    main()