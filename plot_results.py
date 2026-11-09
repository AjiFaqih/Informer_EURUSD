"""Create evaluation plots from saved Informer CSV outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR, PLOTS_DIR, PRED_LENS

METRICS = {
    "test_close_MAE": "Test MAE Close",
    "test_close_MSE": "Test MSE Close",
    "test_close_MAPE_percent": "Test MAPE Close (%)",
    "test_close_DA_percent": "Test Directional Accuracy (%)",
    "sample_close_MAE": "Sample MAE Close",
    "sample_close_MSE": "Sample MSE Close",
    "sample_close_MAPE_percent": "Sample MAPE Close (%)",
    "sample_close_DA_percent": "Sample Directional Accuracy (%)",
}
SEQ_LINESTYLES = {48: "-", 96: "--", 192: "-.", 384: ":"}
SEQ_MARKERS = {48: "o", 96: "s", 192: "^", 384: "x"}


def plot_metric_comparisons(results: pd.DataFrame) -> None:
    results = results.copy()
    results["scenario_label"] = results.apply(
        lambda row: f"S{int(row['seq_len'])} P{int(row['pred_len'])}", axis=1
    )
    for metric, title in METRICS.items():
        if metric not in results.columns:
            continue
        temp = results.dropna(subset=[metric]).sort_values(
            metric, ascending=("DA" not in metric)
        )
        if temp.empty:
            continue
        plt.figure(figsize=(11, 5))
        plt.barh(temp["scenario_label"], temp[metric])
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel("Konfigurasi")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        out = PLOTS_DIR / f"metric_{metric}.png"
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        print("Saved:", out)


def plot_loss_curves() -> None:
    loss_path = RESULTS_DIR / "informer_loss_history.csv"
    if not loss_path.exists():
        print("Lewati plot loss: file riwayat loss belum tersedia.")
        return
    loss = pd.read_csv(loss_path)
    if loss.empty:
        print("Lewati plot loss: file riwayat loss kosong.")
        return
    required = {"seq_len", "pred_len", "epoch", "train_loss", "val_loss"}
    if not required.issubset(loss.columns):
        print("Lewati plot loss: kolom yang dibutuhkan tidak lengkap.")
        return
    for (seq_len, pred_len), group in loss.groupby(["seq_len", "pred_len"], sort=True):
        group = group.sort_values("epoch")
        plt.figure(figsize=(8, 4.5))
        plt.plot(group["epoch"], group["train_loss"], marker="o", label="Train loss")
        plt.plot(group["epoch"], group["val_loss"], marker="o", label="Validation loss")
        best_idx = group["val_loss"].idxmin()
        best = group.loc[best_idx]
        plt.axvline(
            best["epoch"], linestyle="--", linewidth=1,
            label=f"Best epoch={int(best['epoch'])}"
        )
        plt.title(f"Informer Loss | SEQ_LEN={int(seq_len)} | PRED_LEN={int(pred_len)}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss (scaled log_return)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = PLOTS_DIR / f"loss_seq{int(seq_len)}_pred{int(pred_len)}.png"
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        print("Saved:", out)


def plot_100_window_comparisons() -> None:
    """Plot actual Close and each SEQ_LEN's prediction on the final horizon step."""
    prediction_path = RESULTS_DIR / "informer_sample_predictions.csv"
    if not prediction_path.exists():
        print(
            "Lewati plot perbandingan prediksi: informer_sample_predictions.csv belum ada. "
            "Jalankan eksperimen terlebih dahulu."
        )
        return

    predictions = pd.read_csv(prediction_path, parse_dates=["date"])
    if predictions.empty:
        print("Lewati plot perbandingan prediksi: file prediksi sampel kosong.")
        return
    required = {
        "seq_len", "pred_len", "forecast_step", "prediction_order",
        "true_close", "pred_close",
    }
    if not required.issubset(predictions.columns):
        missing = sorted(required - set(predictions.columns))
        raise ValueError(f"Kolom prediksi sampel tidak lengkap: {missing}")

    # Menyamakan definisi dengan notebook: untuk PRED_LEN>1, gunakan horizon terakhir.
    # Dengan demikian, setiap garis berisi 100 prediksi (satu untuk setiap origin).
    for pred_len in PRED_LENS:
        subset = predictions[
            (predictions["pred_len"] == pred_len)
            & (predictions["forecast_step"] == pred_len)
        ].copy()
        if subset.empty:
            print(f"Lewati grafik PRED_LEN={pred_len}: hasil prediksi belum tersedia.")
            continue

        subset = subset.sort_values(["seq_len", "prediction_order"])
        actual_source = subset[subset["seq_len"] == subset["seq_len"].min()]
        actual = (
            actual_source[["prediction_order", "true_close"]]
            .drop_duplicates(subset=["prediction_order"])
            .sort_values("prediction_order")
        )

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(
            actual["prediction_order"], actual["true_close"],
            color="tab:blue", linewidth=2.0, label="Close aktual", zorder=5
        )

        seqs_plotted = []
        for seq_len, group in subset.groupby("seq_len", sort=True):
            group = group.sort_values("prediction_order")
            seq_len = int(seq_len)
            ax.plot(
                group["prediction_order"], group["pred_close"],
                linestyle=SEQ_LINESTYLES.get(seq_len, "-"),
                marker=SEQ_MARKERS.get(seq_len),
                markevery=max(1, len(group) // 10),
                markersize=3,
                linewidth=1.25,
                label=f"Prediksi LR→Close S{seq_len}",
            )
            seqs_plotted.append(seq_len)

        ax.set_title(f"100 prediksi sampel | LR→Close | PRED_LEN={pred_len}")
        ax.set_xlabel("Urutan prediksi")
        ax.set_ylabel("Close EUR/USD")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(100, int(actual["prediction_order"].max())))
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
        fig.tight_layout()
        fig.subplots_adjust(right=0.79)

        out = PLOTS_DIR / f"comparison_100_windows_pred{pred_len}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out} | SEQ_LEN yang tersedia: {seqs_plotted}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "informer_results.csv"
    if result_path.exists():
        results = pd.read_csv(result_path)
        if "error" in results.columns:
            results = results[results["error"].fillna("").eq("")].copy()
        if not results.empty:
            plot_metric_comparisons(results)
        else:
            print("Lewati plot metrik: belum ada skenario yang berhasil.")
    else:
        print("Lewati plot metrik: informer_results.csv belum ada.")

    plot_loss_curves()
    plot_100_window_comparisons()
    print(f"Plot tersimpan di: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
