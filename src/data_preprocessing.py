"""
data_preprocessing.py - Script preprocessing data
Membaca data mentah, membuat fitur biner, split train/test, dan menyimpan hasilnya.
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Tambahkan parent dir ke path agar bisa import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_raw_data, DATA_PROCESSED_DIR, TARGET_COL, DROPOUT_COL


def preprocess():
    """Pipeline preprocessing utama."""
    # 1. Load data mentah
    print("📂 Memuat data mentah...")
    df = load_raw_data()
    print(f"   Shape: {df.shape}")

    # 2. Bersihkan nama kolom (hapus tab/whitespace)
    df.columns = [c.strip().replace('\t', '') for c in df.columns]

    # 3. Cek missing values dan duplikat
    print(f"\n🔍 Missing values total: {df.isnull().sum().sum()}")
    dup = df.duplicated().sum()
    print(f"🔍 Baris duplikat: {dup}")
    if dup > 0:
        df = df.drop_duplicates()
        print(f"   → Duplikat dihapus. Shape baru: {df.shape}")

    # 4. Buat target biner (1 = Dropout, 0 = Non-Dropout)
    print(f"\n🎯 Distribusi Target asli:")
    print(df[TARGET_COL].value_counts().to_string())
    df[DROPOUT_COL] = df[TARGET_COL].apply(lambda x: 1 if x == 'Dropout' else 0)
    print(f"\n🎯 Distribusi dropout_risk:")
    print(df[DROPOUT_COL].value_counts().to_string())

    # 5. Pisahkan fitur dan target
    X = df.drop(columns=[TARGET_COL, DROPOUT_COL])
    y = df[DROPOUT_COL]

    # 6. Train-test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✂️  Train: {X_train.shape}, Test: {X_test.shape}")

    # 7. Simpan ke data/processed/
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(DATA_PROCESSED_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_PROCESSED_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_PROCESSED_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_PROCESSED_DIR, 'y_test.csv'), index=False)

    # Simpan juga full processed data
    df.to_csv(os.path.join(DATA_PROCESSED_DIR, 'data_processed.csv'), index=False)

    print(f"\n💾 Data processed disimpan ke: {DATA_PROCESSED_DIR}")
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    preprocess()
