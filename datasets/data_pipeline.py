"""Data loading, chronological splits, scaling, and Informer window datasets."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_PATH, DATE_COL, CLOSE_COL, LOG_RETURN_COL, RSI_COL, ATR_COL, FEATURE_COLS,
    USE_TIME_FEATURES, MODELING_START, SAMPLE_PREDICTION_END,
    SAMPLE_TARGET_PREDICTIONS, SAMPLE_ALIGNMENT_SEQ_LEN, PRED_LENS,
    TRAIN_RATIO, VAL_RATIO, NUM_WORKERS,
)


class StandardScalerNP:
    """Small NumPy scaler matching the implementation in the provided notebook."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, values: np.ndarray):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return (values - self.mean_) / self.std_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return values * self.std_ + self.mean_


def load_processed_data(path=DATA_PATH) -> pd.DataFrame:
    """Read and validate the processed CSV using the notebook's required columns."""
    path = str(path)
    df = pd.read_csv(path)
    required_cols = [DATE_COL, "Open", "High", "Low", CLOSE_COL, "Volume",
                     LOG_RETURN_COL, RSI_COL, ATR_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}. Kolom yang tersedia: {list(df.columns)}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    for col in required_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[required_cols[1:]] = df[required_cols[1:]].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=required_cols).sort_values(DATE_COL).reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset kosong setelah membersihkan tanggal/nilai numerik yang tidak valid.")
    if (df[CLOSE_COL] <= 0).any():
        raise ValueError("Seluruh nilai Close harus positif untuk rekonstruksi harga dari log return.")
    if df[DATE_COL].duplicated().any():
        n_dup = int(df[DATE_COL].duplicated().sum())
        raise ValueError(f"Ditemukan {n_dup} timestamp duplikat pada kolom {DATE_COL}.")
    df.attrs["dropped_rows"] = before - len(df)
    return df


def make_time_features(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return np.stack([
        d.dt.month.values,
        d.dt.day.values,
        d.dt.weekday.values,
        d.dt.hour.values,
    ], axis=1).astype(np.int64)


def required_reserved_rows() -> int:
    # Keep 100 forecast origins aligned across all SEQ_LEN/PRED_LEN scenarios.
    return SAMPLE_ALIGNMENT_SEQ_LEN + max(PRED_LENS) + SAMPLE_TARGET_PREDICTIONS - 1


def resolve_effective_modeling_end(full_df: pd.DataFrame):
    dates = pd.to_datetime(full_df[DATE_COL]).reset_index(drop=True)
    requested_end = pd.Timestamp(SAMPLE_PREDICTION_END) if SAMPLE_PREDICTION_END else dates.iloc[-1]
    valid_positions = np.flatnonzero((dates <= requested_end).to_numpy())
    if len(valid_positions) == 0:
        raise ValueError("Tidak ada data pada atau sebelum SAMPLE_PREDICTION_END.")
    sample_end_idx = int(valid_positions[-1])
    sample_start_idx = sample_end_idx - required_reserved_rows() + 1
    model_end_idx = sample_start_idx - 1
    if model_end_idx < 0:
        raise ValueError("Dataset terlalu pendek untuk 100 prediksi dan konteks SEQ_LEN maksimum.")
    return dates.iloc[model_end_idx], {
        "sample_start_idx": sample_start_idx,
        "sample_end_idx": sample_end_idx,
        "model_end_idx": model_end_idx,
        "reserved_rows_needed": required_reserved_rows(),
        "first_aligned_forecast_idx": sample_start_idx + SAMPLE_ALIGNMENT_SEQ_LEN,
    }


def split_indices(full_df: pd.DataFrame, effective_end: pd.Timestamp):
    dates = pd.to_datetime(full_df[DATE_COL]).reset_index(drop=True)
    model_positions = np.flatnonzero(((dates >= pd.Timestamp(MODELING_START)) & (dates <= effective_end)).to_numpy())
    if len(model_positions) == 0:
        raise ValueError("Tidak ditemukan baris pada periode modeling.")
    model_start = int(model_positions[0])
    model_end = int(model_positions[-1] + 1)
    n_model = model_end - model_start
    train_end = model_start + int(n_model * TRAIN_RATIO)
    val_end = model_start + int(n_model * (TRAIN_RATIO + VAL_RATIO))
    sample_positions = np.flatnonzero((dates > effective_end).to_numpy())
    if SAMPLE_PREDICTION_END is not None:
        sample_positions = np.flatnonzero(((dates > effective_end) & (dates <= pd.Timestamp(SAMPLE_PREDICTION_END))).to_numpy())
    sample_start = int(sample_positions[0]) if len(sample_positions) else None
    sample_end = int(sample_positions[-1] + 1) if len(sample_positions) else None
    return {"model_start": model_start, "model_end": model_end, "train_end": train_end,
            "val_end": val_end, "sample_start": sample_start, "sample_end": sample_end}


class InformerTimeSeriesDataset(Dataset):
    """Returns encoder inputs, decoder targets/marks and true future price/return."""
    def __init__(self, feature_scaled, target_scaled, time_markers, close_values,
                 log_return_values, seq_len, label_len, pred_len):
        self.feature_scaled = np.asarray(feature_scaled, dtype=np.float32)
        self.target_scaled = np.asarray(target_scaled, dtype=np.float32)
        self.time_markers = np.asarray(time_markers, dtype=np.int64)
        self.close_values = np.asarray(close_values, dtype=np.float32)
        self.log_return_values = np.asarray(log_return_values, dtype=np.float32)
        self.seq_len, self.label_len, self.pred_len = int(seq_len), int(label_len), int(pred_len)

    def __len__(self):
        return max(0, len(self.feature_scaled) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            torch.from_numpy(self.feature_scaled[s_begin:s_end]),
            torch.from_numpy(self.target_scaled[r_begin:r_end]),
            torch.from_numpy(self.time_markers[s_begin:s_end]),
            torch.from_numpy(self.time_markers[r_begin:r_end]),
            torch.tensor(self.close_values[s_end - 1], dtype=torch.float32),
            torch.from_numpy(self.close_values[s_end:s_end + self.pred_len].astype(np.float32)),
            torch.from_numpy(self.log_return_values[s_end:s_end + self.pred_len].astype(np.float32)),
        )


def _make_loader(dataset, batch_size, shuffle=False, drop_last=False):
    if dataset is None or len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=NUM_WORKERS)


def build_dataloaders(frame: pd.DataFrame, seq_len: int, label_len: int, pred_len: int, batch_size: int):
    """Build chronological train/val/test datasets plus an aligned 100-origin sample."""
    target_cols = [LOG_RETURN_COL]
    effective_end, split_info = resolve_effective_modeling_end(frame)
    idx = split_indices(frame, effective_end)
    dates = pd.to_datetime(frame[DATE_COL]).reset_index(drop=True)
    feature_values = frame[FEATURE_COLS].to_numpy(dtype=np.float32)
    target_values = frame[target_cols].to_numpy(dtype=np.float32)
    close_values = frame[CLOSE_COL].to_numpy(dtype=np.float32)
    log_return_values = frame[LOG_RETURN_COL].to_numpy(dtype=np.float32)
    time_markers = make_time_features(frame[DATE_COL]) if USE_TIME_FEATURES else np.zeros((len(frame), 4), dtype=np.int64)

    model_start, train_end, val_end, model_end = idx["model_start"], idx["train_end"], idx["val_end"], idx["model_end"]
    feature_scaler = StandardScalerNP().fit(feature_values[model_start:train_end])
    target_scaler = StandardScalerNP().fit(target_values[model_start:train_end])
    feature_scaled = feature_scaler.transform(feature_values)
    target_scaled = target_scaler.transform(target_values)

    train_slice = slice(model_start, train_end)
    val_slice = slice(max(model_start, train_end - seq_len), val_end)
    test_slice = slice(max(model_start, val_end - seq_len), model_end)

    sample_slice = None
    sample_first_forecast_idx = None
    sample_count = 0
    if idx["sample_start"] is not None:
        sample_start, sample_end = idx["sample_start"], idx["sample_end"]
        sample_first_forecast_idx = sample_start + SAMPLE_ALIGNMENT_SEQ_LEN
        sample_slice_start = sample_first_forecast_idx - seq_len
        sample_count_available = sample_end - sample_first_forecast_idx - pred_len + 1
        sample_count = min(sample_count_available, SAMPLE_TARGET_PREDICTIONS)
        if sample_slice_start >= sample_start and sample_count > 0:
            sample_slice_end = sample_first_forecast_idx + pred_len + sample_count - 1
            sample_slice = slice(sample_slice_start, sample_slice_end)
        else:
            sample_slice = None
            sample_first_forecast_idx = None
            sample_count = 0

    def _ds(s):
        return InformerTimeSeriesDataset(feature_scaled[s], target_scaled[s], time_markers[s],
                                         close_values[s], log_return_values[s], seq_len, label_len, pred_len)

    train_ds = _ds(train_slice)
    val_ds = _ds(val_slice)
    test_ds = _ds(test_slice)
    sample_ds = _ds(sample_slice) if sample_slice is not None else None
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"Baris tidak cukup untuk SEQ_LEN={seq_len}, PRED_LEN={pred_len}.")

    meta = {
        "feature_cols": list(FEATURE_COLS), "target_cols": target_cols,
        "enc_in": len(FEATURE_COLS), "dec_in": len(target_cols), "c_out": len(target_cols),
        "use_time_features": USE_TIME_FEATURES,
        "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
        "sample_size": 0 if sample_ds is None else len(sample_ds),
        "sample_first_forecast_idx": sample_first_forecast_idx,
        "sample_prediction_count": int(sample_count),
        "modeling_start": dates.iloc[model_start], "modeling_end": dates.iloc[model_end - 1],
        "sample_start": None if idx["sample_start"] is None else dates.iloc[idx["sample_start"]],
        "sample_end": None if idx["sample_end"] is None else dates.iloc[idx["sample_end"] - 1],
        "first_sample_forecast_date": None if sample_first_forecast_idx is None else dates.iloc[sample_first_forecast_idx],
        "reserved_rows_needed": split_info["reserved_rows_needed"],
    }
    return (
        _make_loader(train_ds, batch_size, shuffle=True, drop_last=True),
        _make_loader(val_ds, batch_size),
        _make_loader(test_ds, batch_size),
        _make_loader(sample_ds, batch_size),
        feature_scaler, target_scaler, meta,
    )
