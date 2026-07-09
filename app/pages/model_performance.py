import streamlit as st
import json
import os
import pandas as pd

st.title("📈 Model Performance")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

if not os.path.exists(METRICS_PATH):
    st.error(f"File metrics tidak ditemukan: {METRICS_PATH}")
    st.stop()

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

st.subheader("1. Model yang Digunakan")
st.write(metrics.get("model_name", "Tidak diketahui"))

st.subheader("2. Hasil Evaluasi")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")

with col2:
    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")

with col3:
    st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")

# optional jika classification report disimpan
if "classification_report" in metrics:
    st.subheader("3. Classification Report")
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    st.dataframe(report_df)

# optional jika confusion matrix disimpan
if "confusion_matrix" in metrics:
    st.subheader("4. Confusion Matrix")
    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        columns=["Pred 0", "Pred 1"],
        index=["Actual 0", "Actual 1"]
    )
    st.dataframe(cm)