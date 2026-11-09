# EURUSD_Informer

Proyek ini memindahkan implementasi Informer dari notebook `INFORMER_EURUSD_Main(1).ipynb` menjadi skrip Python dengan struktur folder modular. Arsitektur Informer yang diambil dari notebook tetap menggunakan ProbSparse attention, temporal embedding, distillation, serta konfigurasi inti yang sama.

## Struktur

```text
EURUSD_Informer/
├── data/EURUSD60_processed.csv
├── datasets/__init__.py
├── datasets/data_pipeline.py
├── models/__init__.py
├── models/informer.py
├── utils/__init__.py
├── utils/metrics.py
├── checkpoints/
├── results/
├── plots/
├── config.py
├── train.py
├── run_all.py
├── plot_results.py
├── requirements.txt
├── INFORMER_EURUSD_Main_DA.ipynb
└── README.md
```

## Isi eksperimen

- Target: `log_return` yang direkonstruksi menjadi harga `Close`.
- Fitur: `Open`, `High`, `Low`, `Close`, `Volume`, `log_return`, `RSI_14`, `ATR_14`.
- Time features: bulan, tanggal, hari dalam minggu, dan jam.
- `SEQ_LEN`: 48, 96, 192, 384.
- `PRED_LEN`: 1, 4, 8.
- 100 forecast origins pada sampel khusus, disejajarkan antar-konfigurasi menggunakan konteks maksimum `SEQ_LEN=384`.
- Split kronologis pada periode modeling: train 70%, validation 15%, test 15%; data sampel akhir dipisahkan dari periode modeling sebagaimana dilakukan notebook.
- Hyperparameter arsitektur, batch size, learning rate, jumlah epoch, dan early stopping diatur di `config.py`.

## Metrik

`results/informer_results.csv` berisi MAE, MSE, MAPE (%), dan Directional Accuracy (DA, %).

- MAE, MSE, dan MAPE dihitung dari harga `Close` hasil rekonstruksi dari prediksi log return.
- DA dihitung dari kesamaan tanda predicted `log_return` dengan actual `log_return`, untuk seluruh horizon yang dievaluasi: `mean(sign(predicted_return) == sign(actual_return)) * 100`.
- DA tersedia sebagai `test_close_DA_percent` dan `sample_close_DA_percent`. Nama kolom mengikuti bentuk metrik Close lainnya, meskipun DA mengukur arah return.
- Untuk `PRED_LEN > 1`, semua langkah horizon masuk dalam DA dan metrik harga. Forecast origin saling tumpang tindih, sesuai evaluasi rolling window.

## Instalasi

Gunakan Python 3.10–3.13 untuk kompatibilitas umum dengan ekosistem PyTorch saat ini; pada Python 3.14, pastikan wheel PyTorch untuk versi Anda benar-benar tersedia.

```powershell
python -m pip install -r requirements.txt
```

### CUDA

Perintah di bawah memilih CUDA bila PyTorch mendeteksinya dan selain itu memakai CPU:

```powershell
python run_all.py --device auto
```

Untuk meminta CUDA secara eksplisit (akan gagal dengan pesan jelas apabila CUDA tidak tersedia):

```powershell
python run_all.py --device cuda
```

Jalankan satu konfigurasi:

```powershell
python train.py --seq-len 48 --pred-len 1 --device auto
```

Smoke test singkat dengan dua epoch:

```powershell
python train.py --seq-len 48 --pred-len 1 --epochs 2 --device auto
```

Jalankan seluruh 12 konfigurasi (50 epoch maksimum per konfigurasi, early stopping patience 5):

```powershell
python run_all.py --device auto
```

Grafik akan dibuat otomatis di akhir `run_all.py`. Untuk membuat ulang grafik tanpa training:

```powershell
python plot_results.py
```

## Output

- `checkpoints/*.pt`: best weights, config, feature scaler, target scaler.
- `results/informer_results.csv`: satu baris per konfigurasi.
- `results/informer_loss_history.csv`: train/validation loss.
- `results/informer_sample_predictions.csv`: hingga 100 rolling origins × setiap langkah horizon × skenario, termasuk predicted/actual close dan log return.
- `plots/metric_*.png`: grafik perbandingan metrik, termasuk DA.
- `plots/loss_seq*_pred*.png`: kurva train/validation loss tiap skenario.
- `plots/comparison_100_windows_pred1.png`, `comparison_100_windows_pred4.png`, dan `comparison_100_windows_pred8.png`: Close aktual dibanding prediksi Informer untuk `SEQ_LEN=48, 96, 192, 384`.

Grafik perbandingan mengambil `forecast_step=PRED_LEN` (horizon terakhir) untuk setiap rolling origin, sehingga ada hingga 100 titik aktual dan satu garis prediksi untuk setiap nilai `SEQ_LEN`. Grafik hanya memakai hasil eksperimen yang telah tersimpan; jalankan `python run_all.py` terlebih dahulu jika CSV prediksi belum tersedia.

## Catatan metodologis

- Port ini mempertahankan konfigurasi sumber notebook, yaitu split 70%/15%/15% dan `Close` sebagai salah satu fitur input historis. Proyek LSTM sebelumnya memakai split 70%/10%/20% dan tidak memasukkan `Close` ke `FEATURE_COLS`. Untuk perbandingan LSTM-vs-Informer yang benar-benar setara dalam skripsi, samakan rasio split dan daftar fitur input di kedua `config.py` sebelum menjalankan eksperimen final.

1. Fitur dan target di-scaling menggunakan statistik dari bagian train saja.
2. Baris yang dialokasikan untuk sampel prediksi akhir tidak masuk ke periode training/validation/test.
3. Konteks pada awal validation dan test dapat menggunakan baris dari split sebelumnya, tetapi forecast target berada seluruhnya di split yang sedang dinilai.
4. DA dihitung dari arah log return, bukan arah perubahan nilai `Close` hasil prediksi kumulatif.
5. `run_all.py` menyimpan hasil setelah setiap skenario agar progres tetap tersedia jika eksekusi terhenti. Baris gagal dicatat di kolom `error`.
