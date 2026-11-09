"""Train/evaluate one Informer experiment. Supports --device auto|cpu|cuda."""
from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    DATA_PATH, CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR, TARGET_MODE,
    SEQ_LENS, PRED_LENS, BATCH_SIZE, TRAIN_EPOCHS, LEARNING_RATE,
    PATIENCE, MIN_DELTA, D_MODEL, N_HEADS, E_LAYERS, D_LAYERS, D_FF,
    FACTOR, DROPOUT, ATTN, EMBED, ACTIVATION, DISTIL, OUTPUT_ATTENTION,
    USE_EARLY_STOPPING, RESTORE_BEST_WEIGHTS, LOSS_FUNCTION, SEED,
    MAX_TRAIN_BATCHES, MAX_EVAL_BATCHES,
)
from datasets.data_pipeline import load_processed_data, build_dataloaders, StandardScalerNP
from models.informer import Informer
from utils.metrics import close_metrics, log_returns_to_close

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_device(device_name: str = "auto"):
    """Select execution device and raise instead of silently falling back if CUDA was requested."""
    global DEVICE
    if device_name == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diminta, tetapi torch.cuda.is_available() bernilai False. Periksa driver/GPU dan instalasi PyTorch CUDA.")
        DEVICE = torch.device("cuda")
    elif device_name == "cpu":
        DEVICE = torch.device("cpu")
    else:
        raise ValueError("--device harus bernilai auto, cpu, atau cuda.")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return DEVICE


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = DEVICE.type == "cuda"


@dataclass
class ExperimentConfig:
    target_mode: str
    seq_len: int
    pred_len: int
    label_len: int
    batch_size: int = BATCH_SIZE
    train_epochs: int = TRAIN_EPOCHS
    learning_rate: float = LEARNING_RATE
    patience: int = PATIENCE
    min_delta: float = MIN_DELTA
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    e_layers: int = E_LAYERS
    d_layers: int = D_LAYERS
    d_ff: int = D_FF
    factor: int = FACTOR
    dropout: float = DROPOUT
    attn: str = ATTN
    embed: str = EMBED
    activation: str = ACTIVATION
    distil: bool = DISTIL
    output_attention: bool = OUTPUT_ATTENTION


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, checkpoint_path=None):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.best_loss = np.inf
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            if self.checkpoint_path:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": self.best_epoch, "best_val_loss": self.best_loss,
                            "model_state_dict": self.best_state}, self.checkpoint_path)
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            return True
        if self.checkpoint_path and self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            return True
        return False


def scenario_label(cfg):
    return f"LR→Close | S{cfg.seq_len} | P{cfg.pred_len}"


def checkpoint_path(cfg):
    return CHECKPOINT_DIR / f"best_informer_{cfg.target_mode}_seq{cfg.seq_len}_label{cfg.label_len}_pred{cfg.pred_len}.pt"


def build_model(cfg, meta):
    return Informer(
        enc_in=meta["enc_in"], dec_in=meta["dec_in"], c_out=meta["c_out"],
        seq_len=cfg.seq_len, label_len=cfg.label_len, out_len=cfg.pred_len,
        factor=cfg.factor, d_model=cfg.d_model, n_heads=cfg.n_heads,
        e_layers=cfg.e_layers, d_layers=cfg.d_layers, d_ff=cfg.d_ff,
        dropout=cfg.dropout, attn=cfg.attn, embed=cfg.embed,
        activation=cfg.activation, output_attention=cfg.output_attention,
        distil=cfg.distil, mix=True, use_time_features=meta["use_time_features"],
    ).to(DEVICE)


def move_batch(batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark, base_close, future_close, future_log_return = batch
    return (
        batch_x.float().to(DEVICE), batch_y.float().to(DEVICE),
        batch_x_mark.long().to(DEVICE), batch_y_mark.long().to(DEVICE),
        base_close.float().to(DEVICE), future_close.float().to(DEVICE), future_log_return.float().to(DEVICE),
    )


def run_epoch(model, loader, optimizer, criterion, cfg, train=True):
    model.train(train)
    losses = []
    iterator = enumerate(loader)
    if train:
        iterator = tqdm(iterator, total=len(loader), leave=False, desc="train")
    with torch.set_grad_enabled(train):
        for batch_i, batch in iterator:
            if train and MAX_TRAIN_BATCHES is not None and batch_i >= MAX_TRAIN_BATCHES:
                break
            if (not train) and MAX_EVAL_BATCHES is not None and batch_i >= MAX_EVAL_BATCHES:
                break
            batch_x, batch_y, batch_x_mark, batch_y_mark, *_ = move_batch(batch)
            dec_zeros = torch.zeros_like(batch_y[:, -cfg.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_zeros], dim=1)
            if train:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            target = batch_y[:, -cfg.pred_len:, :]
            loss = criterion(outputs, target)
            if train:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_loader(model, loader, cfg, target_scaler: StandardScalerNP):
    if loader is None:
        return None
    model.eval()
    pred_scaled, true_scaled, base_close, true_close_actual, true_log_actual = [], [], [], [], []
    for batch_i, batch in enumerate(loader):
        if MAX_EVAL_BATCHES is not None and batch_i >= MAX_EVAL_BATCHES:
            break
        batch_x, batch_y, batch_x_mark, batch_y_mark, base, future_close, future_log = move_batch(batch)
        dec_zeros = torch.zeros_like(batch_y[:, -cfg.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_zeros], dim=1)
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        pred_scaled.append(outputs.detach().cpu().numpy())
        true_scaled.append(batch_y[:, -cfg.pred_len:, :].detach().cpu().numpy())
        base_close.append(base.detach().cpu().numpy())
        true_close_actual.append(future_close.detach().cpu().numpy())
        true_log_actual.append(future_log.detach().cpu().numpy())
    if not pred_scaled:
        raise RuntimeError("Evaluation menghasilkan nol batch. Periksa dataset, batch size, dan batas MAX_EVAL_BATCHES.")
    pred_scaled = np.concatenate(pred_scaled, axis=0)
    true_scaled = np.concatenate(true_scaled, axis=0)
    base_close = np.concatenate(base_close, axis=0)
    true_close = np.concatenate(true_close_actual, axis=0)
    true_log_return = np.concatenate(true_log_actual, axis=0)

    pred_target = target_scaler.inverse_transform(pred_scaled)
    true_target = target_scaler.inverse_transform(true_scaled)
    pred_log_return = pred_target[..., 0]
    # Reconstruct the predicted Close using the last observed Close as anchor.
    pred_close = log_returns_to_close(base_close, pred_log_return)
    metrics = close_metrics(true_close, pred_close, true_log_return, pred_log_return)
    return {
        "pred_close": pred_close,
        "true_close": true_close,
        "base_close": base_close,
        "true_log_return": true_log_return,
        "pred_log_return": pred_log_return,
        "metrics": metrics,
    }


def run_experiment(cfg, frame, verbose=True):
    set_seed(SEED)
    train_loader, val_loader, test_loader, sample_loader, feature_scaler, target_scaler, meta = build_dataloaders(
        frame=frame, seq_len=cfg.seq_len, label_len=cfg.label_len,
        pred_len=cfg.pred_len, batch_size=cfg.batch_size,
    )
    model = build_model(cfg, meta)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    ckpt_path = checkpoint_path(cfg)
    early_stop = EarlyStopping(cfg.patience, cfg.min_delta, ckpt_path)
    history = []
    start_time = time.time()
    if verbose:
        print("=" * 84)
        print(scenario_label(cfg))
        print("Device:", DEVICE)
        if DEVICE.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        print("Fitur:", meta["feature_cols"])
        print("Target:", meta["target_cols"])
        print("Periode modeling:", meta["modeling_start"], "sampai", meta["modeling_end"])
        print("Periode sampel:", meta["sample_start"], "sampai", meta["sample_end"])
        print("Prediksi sampel pertama:", meta["first_sample_forecast_date"])
        print("Jumlah window Train/Val/Test/Sampel:", meta["train_size"], meta["val_size"], meta["test_size"], meta["sample_size"])
        print("Jumlah parameter trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(1, cfg.train_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, cfg, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, cfg, train=False)
        stopped = early_stop.step(val_loss, model, epoch) if USE_EARLY_STOPPING else False
        best_loss = early_stop.best_loss if USE_EARLY_STOPPING else val_loss
        best_epoch = early_stop.best_epoch if USE_EARLY_STOPPING else epoch
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "best_val_loss": best_loss, "best_epoch": best_epoch})
        if verbose:
            print(f"Epoch {epoch:03d} | train={train_loss:.7f} | val={val_loss:.7f} | best_val={best_loss:.7f}")
        if stopped:
            if verbose:
                print(f"Early stopping pada epoch {epoch}; epoch terbaik={early_stop.best_epoch}")
            break
    if USE_EARLY_STOPPING and RESTORE_BEST_WEIGHTS:
        early_stop.restore(model)

    history_df = pd.DataFrame(history)
    if history_df.empty or not np.isfinite(history_df["val_loss"]).any():
        raise RuntimeError("Training tidak menghasilkan validation loss yang valid.")
    test_pred = predict_loader(model, test_loader, cfg, target_scaler)
    sample_pred = predict_loader(model, sample_loader, cfg, target_scaler) if sample_loader is not None else None
    best_idx = history_df["val_loss"].idxmin()
    result = {
        "scenario_label": scenario_label(cfg), "target_mode": cfg.target_mode,
        "features": ",".join(meta["feature_cols"]), "target_cols": ",".join(meta["target_cols"]),
        "loss_function": LOSS_FUNCTION, "seq_len": cfg.seq_len, "label_len": cfg.label_len,
        "pred_len": cfg.pred_len, "best_epoch": int(history_df.loc[best_idx, "epoch"]),
        "best_val_loss": float(history_df.loc[best_idx, "val_loss"]),
        "epochs_ran": len(history_df), "elapsed_seconds": time.time() - start_time,
        "checkpoint_path": str(ckpt_path), "device": str(DEVICE),
        "modeling_start": str(meta["modeling_start"]), "modeling_end": str(meta["modeling_end"]),
        "sample_start": str(meta["sample_start"]), "sample_end": str(meta["sample_end"]),
        "sample_first_forecast_date": str(meta["first_sample_forecast_date"]),
        "sample_prediction_count": meta["sample_prediction_count"],
        **{f"test_{k}": v for k, v in test_pred["metrics"].items()},
    }
    if sample_pred is not None:
        result.update({f"sample_{k}": v for k, v in sample_pred["metrics"].items()})
    history_df["target_mode"] = cfg.target_mode
    history_df["seq_len"] = cfg.seq_len
    history_df["label_len"] = cfg.label_len
    history_df["pred_len"] = cfg.pred_len
    history_df["scenario_label"] = scenario_label(cfg)
    history_df["loss_function"] = LOSS_FUNCTION

    # Save best weights plus scalers and full run configuration for reproducibility.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": result["best_epoch"], "best_val_loss": result["best_val_loss"],
        "model_state_dict": model.state_dict(), "config": asdict(cfg),
        "feature_scaler_mean": feature_scaler.mean_, "feature_scaler_std": feature_scaler.std_,
        "target_scaler_mean": target_scaler.mean_, "target_scaler_std": target_scaler.std_,
        "feature_cols": meta["feature_cols"], "target_cols": meta["target_cols"],
    }, ckpt_path)
    return model, history_df, result, test_pred, sample_pred, meta


def make_sample_rows(sample_pred, cfg, meta, frame):
    if sample_pred is None:
        return []
    first_idx = int(meta["sample_first_forecast_idx"])
    rows = []
    for i in range(sample_pred["pred_close"].shape[0]):
        for step in range(cfg.pred_len):
            global_idx = first_idx + i + step
            rows.append({
                "date": frame["date"].iloc[global_idx], "target_mode": cfg.target_mode,
                "seq_len": cfg.seq_len, "label_len": cfg.label_len, "pred_len": cfg.pred_len,
                "scenario_label": scenario_label(cfg), "prediction_order": i + 1, "forecast_step": step + 1,
                "base_close": float(sample_pred["base_close"][i]),
                "true_close": float(sample_pred["true_close"][i, step]),
                "pred_close": float(sample_pred["pred_close"][i, step]),
                "true_log_return": float(sample_pred["true_log_return"][i, step]),
                "pred_log_return": float(sample_pred["pred_log_return"][i, step]),
            })
    return rows


def run_one(seq_len: int, pred_len: int, device: str = "auto", epochs: int | None = None):
    configure_device(device)
    frame = load_processed_data(DATA_PATH)
    cfg = ExperimentConfig(target_mode=TARGET_MODE, seq_len=int(seq_len), pred_len=int(pred_len),
                          label_len=int(seq_len) // 2, train_epochs=epochs or TRAIN_EPOCHS)
    _, hist, row, _test_pred, sample_pred, meta = run_experiment(cfg, frame, verbose=True)
    row["error"] = ""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = RESULTS_DIR / "informer_results.csv"
    old = pd.read_csv(existing) if existing.exists() else pd.DataFrame()
    if not old.empty:
        old = old[~((old["seq_len"] == seq_len) & (old["pred_len"] == pred_len))]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(existing, index=False)
    hist_path = RESULTS_DIR / "informer_loss_history.csv"
    hist_old = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    if not hist_old.empty:
        hist_old = hist_old[~((hist_old["seq_len"] == seq_len) & (hist_old["pred_len"] == pred_len))]
    pd.concat([hist_old, hist], ignore_index=True).to_csv(hist_path, index=False)
    rows = make_sample_rows(sample_pred, cfg, meta, frame)
    pred_path = RESULTS_DIR / "informer_sample_predictions.csv"
    pred_old = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    if not pred_old.empty:
        pred_old = pred_old[~((pred_old["seq_len"] == seq_len) & (pred_old["pred_len"] == pred_len))]
    if rows:
        pd.concat([pred_old, pd.DataFrame(rows)], ignore_index=True).to_csv(pred_path, index=False)
    print(f"Hasil metrik: {existing}")
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, choices=SEQ_LENS, required=True)
    parser.add_argument("--pred-len", type=int, choices=PRED_LENS, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch untuk smoke test; default 50")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_one(args.seq_len, args.pred_len, args.device, args.epochs)
