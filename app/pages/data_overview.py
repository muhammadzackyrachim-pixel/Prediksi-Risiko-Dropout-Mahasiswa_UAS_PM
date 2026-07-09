import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.title("📊 Data Overview")

# path data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")

if not os.path.exists(DATA_PATH):
    st.error(f"File tidak ditemukan: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.subheader("1. Preview Dataset")
st.dataframe(df.head())

st.subheader("2. Dimensi Dataset")
st.write(f"Jumlah baris: **{df.shape[0]}**")
st.write(f"Jumlah kolom: **{df.shape[1]}**")

st.subheader("3. Tipe Data")
dtype_df = pd.DataFrame({
    "Kolom": df.columns,
    "Tipe Data": df.dtypes.astype(str)
})
st.dataframe(dtype_df)

st.subheader("4. Missing Values")
missing_df = pd.DataFrame({
    "Kolom": df.columns,
    "Jumlah Missing": df.isnull().sum().values
})
st.dataframe(missing_df)

# target distribution
if 'dropout_risk' in df.columns:
    st.subheader("5. Distribusi Target (dropout_risk)")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='dropout_risk', data=df, ax=ax)
    ax.set_title("Distribusi Target Dropout Risk")
    st.pyplot(fig)

st.subheader("6. Statistik Deskriptif")
st.dataframe(df.describe())

# korelasi hanya numeric
st.subheader("7. Heatmap Korelasi")
num_df = df.select_dtypes(include=['int64', 'float64'])

if len(num_df.columns) > 1:
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(num_df.corr(), cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Kolom numerik tidak cukup untuk membuat heatmap.")