"""
utils.py - Fungsi utilitas untuk proyek Prediksi Dropout Mahasiswa
"""
import os
import pandas as pd
import joblib

# ───── Path constants ─────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'data.csv')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ───── Feature lists ─────
NUMERIC_FEATURES = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Previous qualification', 'Previous qualification (grade)',
    'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Admission grade',
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP',
]

# Fitur yang bersifat biner (0/1) - untuk form input Streamlit
BINARY_FEATURES = [
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International',
]

# Label deskriptif untuk fitur di form Streamlit
FEATURE_LABELS = {
    'Marital status': 'Status Pernikahan (1=Single, 2=Married, ...)',
    'Application mode': 'Mode Pendaftaran',
    'Application order': 'Urutan Pendaftaran (0-9)',
    'Course': 'Kode Program Studi',
    'Previous qualification': 'Kualifikasi Sebelumnya',
    'Previous qualification (grade)': 'Nilai Kualifikasi Sebelumnya',
    'Nacionality': 'Kewarganegaraan (kode)',
    "Mother's qualification": 'Pendidikan Ibu',
    "Father's qualification": 'Pendidikan Ayah',
    "Mother's occupation": 'Pekerjaan Ibu',
    "Father's occupation": 'Pekerjaan Ayah',
    'Admission grade': 'Nilai Masuk',
    'Displaced': 'Pengungsi (0=Tidak, 1=Ya)',
    'Educational special needs': 'Kebutuhan Khusus (0=Tidak, 1=Ya)',
    'Debtor': 'Memiliki Hutang (0=Tidak, 1=Ya)',
    'Tuition fees up to date': 'SPP Lunas (0=Tidak, 1=Ya)',
    'Gender': 'Jenis Kelamin (0=Perempuan, 1=Laki-laki)',
    'Scholarship holder': 'Penerima Beasiswa (0=Tidak, 1=Ya)',
    'Age at enrollment': 'Usia Saat Mendaftar',
    'International': 'Mahasiswa Internasional (0=Tidak, 1=Ya)',
    'Curricular units 1st sem (credited)': 'SKS Diakui Sem 1',
    'Curricular units 1st sem (enrolled)': 'SKS Diambil Sem 1',
    'Curricular units 1st sem (evaluations)': 'SKS Dievaluasi Sem 1',
    'Curricular units 1st sem (approved)': 'SKS Lulus Sem 1',
    'Curricular units 1st sem (grade)': 'Rata-rata Nilai Sem 1',
    'Curricular units 1st sem (without evaluations)': 'SKS Tanpa Evaluasi Sem 1',
    'Curricular units 2nd sem (credited)': 'SKS Diakui Sem 2',
    'Curricular units 2nd sem (enrolled)': 'SKS Diambil Sem 2',
    'Curricular units 2nd sem (evaluations)': 'SKS Dievaluasi Sem 2',
    'Curricular units 2nd sem (approved)': 'SKS Lulus Sem 2',
    'Curricular units 2nd sem (grade)': 'Rata-rata Nilai Sem 2',
    'Curricular units 2nd sem (without evaluations)': 'SKS Tanpa Evaluasi Sem 2',
    'Unemployment rate': 'Tingkat Pengangguran (%)',
    'Inflation rate': 'Tingkat Inflasi (%)',
    'GDP': 'GDP',
}

TARGET_COL = 'Target'
DROPOUT_COL = 'dropout_risk'


def load_raw_data() -> pd.DataFrame:
    """Load dataset mentah dari CSV."""
    return pd.read_csv(DATA_RAW, sep=';')


def load_processed_data():
    """Load data yang sudah diproses (X_train, X_test, y_train, y_test)."""
    return (
        pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'X_train.csv')),
        pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'X_test.csv')),
        pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'y_train.csv')).values.ravel(),
        pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'y_test.csv')).values.ravel(),
    )


def save_model(model, filename):
    """Simpan model ke folder models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"✅ Model disimpan: {path}")


def load_model(filename):
    """Load model dari folder models/."""
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)
