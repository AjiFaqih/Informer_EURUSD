"""Run the full 4 x 3 Informer experiment grid."""
import argparse
import numpy as np
import pandas as pd
import torch

from config import (
    DATA_PATH, RESULTS_DIR, CHECKPOINT_DIR, SEQ_LENS, PRED_LENS, TARGET_MODE,
    TRAIN_EPOCHS,
)
from datasets.data_pipeline import load_processed_data
from train import ExperimentConfig, configure_device, run_experiment, make_sample_rows, scenario_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for debugging only")
    args = parser.parse_args()
    configure_device(args.device)
    frame = load_processed_data(DATA_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    result_rows, histories, sample_rows = [], [], []
    total = len(SEQ_LENS) * len(PRED_LENS)
    index = 0
    for seq_len in SEQ_LENS:
        for pred_len in PRED_LENS:
            index += 1
            cfg = ExperimentConfig(target_mode=TARGET_MODE, seq_len=seq_len, pred_len=pred_len,
                                   label_len=seq_len // 2, train_epochs=args.epochs or TRAIN_EPOCHS)
            print("\n" + "=" * 84)
            print(f"Eksperimen {index}/{total}: SEQ_LEN={seq_len}, PRED_LEN={pred_len}")
            print("=" * 84)
            try:
                _model, hist, row, _test_pred, sample_pred, meta = run_experiment(cfg, frame, verbose=True)
                row["error"] = ""
                result_rows.append(row)
                histories.append(hist)
                sample_rows.extend(make_sample_rows(sample_pred, cfg, meta, frame))
            except Exception as exc:
                if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                row = {"scenario_label": scenario_label(cfg), "target_mode": TARGET_MODE,
                       "seq_len": seq_len, "label_len": seq_len // 2, "pred_len": pred_len,
                       "error": repr(exc)}
                result_rows.append(row)
                print("GAGAL:", repr(exc))
            # Checkpoint progress after each scenario so results remain available if interrupted.
            pd.DataFrame(result_rows).to_csv(RESULTS_DIR / "informer_results.csv", index=False)
            if histories:
                pd.concat(histories, ignore_index=True).to_csv(RESULTS_DIR / "informer_loss_history.csv", index=False)
            if sample_rows:
                pd.DataFrame(sample_rows).to_csv(RESULTS_DIR / "informer_sample_predictions.csv", index=False)
    print("\nSemua skenario selesai.")
    result_df = pd.DataFrame(result_rows)
    print(result_df.to_string(index=False))
    failures = result_df["error"].fillna("").ne("").sum() if "error" in result_df else 0
    if failures:
        print(f"Peringatan: {failures}/{total} eksperimen gagal. Lihat kolom error pada results/informer_results.csv.")

    # Automatically generate comparison plots after the experiment grid.
    # The plotting code skips outputs whose corresponding CSV results are unavailable.
    print("\nMembuat plot evaluasi dan perbandingan prediksi...")
    try:
        from plot_results import main as make_plots
        make_plots()
    except Exception as exc:
        print(f"Peringatan: pembuatan plot gagal: {exc!r}")


if __name__ == "__main__":
    main()
