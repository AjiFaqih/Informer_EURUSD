"""Evaluation metrics for price reconstruction and return-direction accuracy."""
import numpy as np


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mse_np(y_true, y_pred):
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(diff ** 2))


def mape_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def log_returns_to_close(base_close, log_returns):
    """Reconstruct each forecast horizon's Close from anchor Close and predicted log returns."""
    log_returns = np.asarray(log_returns)
    if log_returns.ndim == 3:
        log_returns = log_returns[..., 0]
    return np.asarray(base_close).reshape(-1, 1) * np.exp(np.cumsum(log_returns, axis=1))


def directional_accuracy(true_log_return, pred_log_return):
    """Percent of forecasted returns whose sign matches the realized return's sign."""
    actual = np.asarray(true_log_return).reshape(-1)
    predicted = np.asarray(pred_log_return).reshape(-1)
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("Directional Accuracy membutuhkan jumlah true/predicted log_return yang sama dan tidak kosong.")
    return float(np.mean(np.sign(actual) == np.sign(predicted)) * 100.0)


def close_metrics(true_close, pred_close, true_log_return, pred_log_return):
    """MAE/MSE/MAPE are measured on reconstructed Close; DA on log-return signs."""
    return {
        "close_MAE": mae_np(true_close, pred_close),
        "close_MSE": mse_np(true_close, pred_close),
        "close_MAPE_percent": mape_np(true_close, pred_close),
        "close_DA_percent": directional_accuracy(true_log_return, pred_log_return),
    }
