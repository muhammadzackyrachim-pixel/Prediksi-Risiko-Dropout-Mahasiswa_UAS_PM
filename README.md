# Prediksi Risiko Dropout Mahasiswa Menggunakan Machine Learning

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem **prediksi risiko dropout mahasiswa** menggunakan pendekatan **Machine Learning**. Model dikembangkan berdasarkan data akademik, sosial-ekonomi, dan performa mahasiswa untuk membantu mengidentifikasi mahasiswa yang berisiko tidak menyelesaikan studi tepat waktu.

Selain membangun model prediksi, proyek ini juga menyediakan **dashboard interaktif berbasis Streamlit** untuk:
- menampilkan ringkasan dataset,
- melihat performa model,
- dan melakukan prediksi risiko dropout secara langsung.

---

## Latar Belakang
Dropout mahasiswa merupakan salah satu permasalahan penting dalam dunia pendidikan tinggi karena dapat memengaruhi:
- tingkat kelulusan institusi,
- kualitas akademik,
- efisiensi pembelajaran,
- dan akreditasi perguruan tinggi.

Dengan memanfaatkan data historis mahasiswa, model machine learning dapat digunakan untuk mendeteksi pola-pola yang berkaitan dengan risiko dropout sehingga institusi dapat melakukan **intervensi lebih dini**.

---

## Tujuan Proyek
Tujuan dari proyek ini adalah:
1. Melakukan **eksplorasi dan analisis data mahasiswa**.
2. Membangun model machine learning untuk **memprediksi risiko dropout mahasiswa**.
3. Mengevaluasi performa model menggunakan metrik klasifikasi.
4. Menyediakan **aplikasi dashboard interaktif** menggunakan Streamlit untuk visualisasi dan prediksi.

---

## Dataset
Dataset yang digunakan berisi informasi mahasiswa, seperti:
- status pernikahan,
- jalur pendaftaran,
- nilai masuk,
- status beasiswa,
- status pembayaran biaya kuliah,
- jumlah mata kuliah yang diambil/disetujui,
- nilai semester 1 dan semester 2,
- serta label status mahasiswa.

### Target Klasifikasi
Kolom target asli pada dataset adalah:
- `Dropout`
- `Graduate`
- `Enrolled`

Untuk kebutuhan klasifikasi biner, target diubah menjadi kolom baru bernama:

- `dropout_risk = 1` → jika mahasiswa **Dropout**
- `dropout_risk = 0` → jika mahasiswa **bukan Dropout** (`Graduate` / `Enrolled`)

---

## Struktur Proyek
Berikut struktur direktori proyek:

```bash
student-dropout-project/
│
├── app/
│   ├── app.py
│   └── assets/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── feature_importance.png
│       └── style.css
│
├── data/
│   ├── raw/
│   │   └── data.csv
│   │
│   ├── processed/
│   │   ├── processed_data.csv
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   │
│   └── external/
│       └── data_dictionary.md
│
├── figures/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── models/
│   ├── full_pipeline.pkl
│   ├── preprocessor.pkl
│   └── model_metrics.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
│
├── requirements.txt
└── README.md