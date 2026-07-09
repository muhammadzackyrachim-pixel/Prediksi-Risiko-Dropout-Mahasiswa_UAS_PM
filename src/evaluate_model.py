"""
evaluate_model.py - Script evaluasi model
Menghitung metrik performa dan menampilkan classification report.
"""
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_processed_data, load_model


def evaluate():
    """Evaluasi model terbaik pada test set."""
    # 1. Load data & model
    print("📂 Memuat data dan model...")
    X_train, X_test, y_train, y_test = load_processed_data()
    model = load_model('best_model.pkl')

    # 2. Prediksi
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 3. Metrik
    print("\n" + "=" * 50)
    print("📊 EVALUASI MODEL TERBAIK (Random Forest Tuned)")
    print("=" * 50)

    print(f"\n   Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"   F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f"   ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Dropout', 'Dropout']))

    print(f"\n📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }


if __name__ == '__main__':
    evaluate()
