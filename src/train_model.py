"""
train_model.py - Script training model
Melatih Logistic Regression & Random Forest, hyperparameter tuning, dan menyimpan model terbaik.
"""
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_processed_data, save_model, NUMERIC_FEATURES


def build_preprocessor(numeric_features):
    """Bangun ColumnTransformer untuk preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
    ])
    return preprocessor


def train():
    """Pipeline training utama."""
    # 1. Load data
    print("📂 Memuat data processed...")
    X_train, X_test, y_train, y_test = load_processed_data()

    # Deteksi fitur numerik yang tersedia
    numeric_features = [f for f in NUMERIC_FEATURES if f in X_train.columns]
    print(f"   Fitur numerik: {len(numeric_features)}")

    # 2. Build preprocessor
    preprocessor = build_preprocessor(numeric_features)

    # ═══════════════════════════════════
    # 3. Logistic Regression
    # ═══════════════════════════════════
    print("\n🔵 Training Logistic Regression...")
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_lr))
    print(f"   ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")

    # ═══════════════════════════════════
    # 4. Random Forest (default)
    # ═══════════════════════════════════
    print("\n🌲 Training Random Forest...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_rf))
    print(f"   ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")

    # ═══════════════════════════════════
    # 5. Hyperparameter Tuning RF
    # ═══════════════════════════════════
    print("\n⚙️  Hyperparameter Tuning Random Forest (GridSearchCV)...")
    rf_tuning = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 15],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(rf_tuning, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"   Best params: {grid.best_params_}")
    print(f"   Best CV F1:  {grid.best_score_:.4f}")

    y_pred_best = best_model.predict(X_test)
    y_prob_best = best_model.predict_proba(X_test)[:, 1]
    print(f"\n📊 Hasil model terbaik pada Test Set:")
    print(classification_report(y_test, y_pred_best))
    print(f"   ROC-AUC: {roc_auc_score(y_test, y_prob_best):.4f}")

    # ═══════════════════════════════════
    # 6. Simpan model
    # ═══════════════════════════════════
    save_model(best_model, 'best_model.pkl')

    # Simpan juga preprocessor terpisah (untuk Streamlit)
    preprocessor_fitted = best_model.named_steps['preprocessor']
    save_model(preprocessor_fitted, 'preprocessing.pkl')

    # Simpan juga LR untuk perbandingan
    save_model(lr_pipeline, 'logistic_regression.pkl')

    print("\n🎉 Training selesai!")
    return best_model


if __name__ == '__main__':
    train()
