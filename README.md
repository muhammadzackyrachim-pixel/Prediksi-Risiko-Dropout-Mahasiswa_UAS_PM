# 🎓 Prediksi Risiko Dropout Mahasiswa

Capstone Project Data Mining — Memprediksi risiko dropout mahasiswa menggunakan teknik Machine Learning.

## 📋 Deskripsi

Proyek ini menggunakan dataset **Students' Dropout and Academic Success** untuk membangun model prediksi dropout mahasiswa. Model yang digunakan adalah **Random Forest Classifier** dengan hyperparameter tuning via GridSearchCV.

## 📁 Struktur Proyek

```
capstone-project-data-mining/
├── data/
│   ├── raw/                # Data mentah
│   ├── processed/          # Data yang sudah diproses
│   └── external/           # Data referensi eksternal
├── notebooks/
│   ├── 01_eda.ipynb        # EDA dan preprocessing
│   ├── 02_modeling.ipynb   # Pemodelan dan evaluasi
│   └── 03_interpretation.ipynb  # Interpretasi model
├── src/
│   ├── data_preprocessing.py   # Script preprocessing
│   ├── train_model.py          # Script training
│   ├── evaluate_model.py       # Script evaluasi
│   └── utils.py                # Fungsi utilitas
├── models/
│   ├── best_model.pkl          # Model terbaik
│   └── preprocessing.pkl       # Pipeline preprocessing
├── app/
│   ├── app.py              # Aplikasi Streamlit utama
│   ├── pages/              # Halaman tambahan Streamlit
│   └── assets/             # Gambar, CSS, dll.
├── reports/
│   ├── final_report.pdf    # Laporan akhir
│   └── presentation.pptx  # Slide presentasi
├── requirements.txt        # Dependencies
├── README.md               # Dokumentasi proyek
└── .gitignore
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocessing Data
```bash
python src/data_preprocessing.py
```

### 3. Training Model
```bash
python src/train_model.py
```

### 4. Evaluasi Model
```bash
python src/evaluate_model.py
```

### 5. Jalankan Aplikasi Streamlit
```bash
streamlit run app/app.py
```

## 📊 Dataset

- **Sumber**: UCI Machine Learning Repository
- **Jumlah data**: 4.424 mahasiswa
- **Jumlah fitur**: 36 fitur
- **Target**: Dropout, Graduate, Enrolled → diubah jadi biner (Dropout vs Non-Dropout)

## 🤖 Model

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | - | - | - | - | - |
| Random Forest (Tuned) | - | - | - | - | - |

> Metrik akan diisi setelah training selesai.

## 🛠️ Teknologi

- Python 3.12
- Scikit-learn
- Pandas & NumPy
- Streamlit
- Plotly
- Matplotlib & Seaborn
