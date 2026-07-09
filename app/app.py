"""
app.py - Aplikasi Streamlit Utama
Prediksi Risiko Dropout Mahasiswa
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Path setup
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, BASE_DIR)

from src.utils import (
    load_raw_data, load_model, NUMERIC_FEATURES,
    BINARY_FEATURES, FEATURE_LABELS, MODELS_DIR, DATA_PROCESSED_DIR
)

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="🎓 Prediksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════
def load_css():
    css_path = os.path.join(APP_DIR, 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Inline critical CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        border: 1px solid #e8ecf4;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .prediction-box {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }
    .pred-dropout {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    .pred-safe {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
    }
    .pred-text {
        font-size: 1.8rem;
        font-weight: 800;
    }
    .pred-prob {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


load_css()

# ═══════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════
@st.cache_data
def get_raw_data():
    df = load_raw_data()
    df.columns = [c.strip().replace('\t', '') for c in df.columns]
    df['dropout_risk'] = df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)
    return df


@st.cache_resource
def get_model():
    try:
        return load_model('best_model.pkl')
    except Exception:
        return None


@st.cache_data
def get_processed_data():
    try:
        X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'y_test.csv')).values.ravel()
        return X_test, y_test
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Navigation")
    st.markdown("---")
    page = st.radio(
        "Pilih Halaman:",
        ["🏠 Home", "📊 EDA Dashboard", "🤖 Prediksi", "📈 Evaluasi Model"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; opacity:0.6; font-size:0.8rem;'>
            <p>Capstone Project Data Mining</p>
            <p>Prediksi Dropout Mahasiswa</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        """
        <div class="main-header">
            <h1>🎓 Prediksi Risiko Dropout Mahasiswa</h1>
            <p>Capstone Project Data Mining — Menggunakan Machine Learning untuk memprediksi risiko dropout mahasiswa</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = get_raw_data()

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Total Mahasiswa</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-label">Total Fitur</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        dropout_pct = (df['Target'] == 'Dropout').mean() * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{dropout_pct:.1f}%</div>
                <div class="metric-label">Dropout Rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">2</div>
                <div class="metric-label">Model Dilatih</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset overview
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("### 📋 Distribusi Status Mahasiswa")
        target_counts = df['Target'].value_counts().reset_index()
        target_counts.columns = ['Status', 'Jumlah']
        colors = {'Graduate': '#00b894', 'Dropout': '#e74c3c', 'Enrolled': '#fdcb6e'}
        fig = px.pie(
            target_counts, values='Jumlah', names='Status',
            color='Status', color_discrete_map=colors,
            hole=0.45,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          textfont_size=14)
        fig.update_layout(
            font=dict(family="Inter"), showlegend=True,
            height=380, margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### 📊 Distribusi Usia Saat Pendaftaran")
        fig2 = px.histogram(
            df, x='Age at enrollment', color='Target',
            color_discrete_map=colors,
            barmode='overlay', opacity=0.7,
            nbins=30,
        )
        fig2.update_layout(
            font=dict(family="Inter"),
            xaxis_title="Usia", yaxis_title="Jumlah",
            height=380, margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Dataset preview
    st.markdown("### 🗂️ Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True, height=350)


# ═══════════════════════════════════════════════════════
# PAGE: EDA DASHBOARD
# ═══════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.markdown(
        """
        <div class="main-header">
            <h1>📊 Exploratory Data Analysis</h1>
            <p>Visualisasi dan analisis data mahasiswa secara mendalam</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = get_raw_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distribusi Fitur", "🔥 Korelasi", "📦 Box Plot", "📊 Perbandingan"
    ])

    # ── Tab 1: Distribusi ──
    with tab1:
        st.markdown("#### Pilih fitur untuk melihat distribusi")
        selected_feat = st.selectbox(
            "Fitur:", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index('Admission grade')
        )
        colors_map = {'Graduate': '#00b894', 'Dropout': '#e74c3c', 'Enrolled': '#fdcb6e'}
        fig = px.histogram(
            df, x=selected_feat, color='Target',
            color_discrete_map=colors_map,
            barmode='overlay', opacity=0.75, nbins=40,
            marginal='box',
        )
        fig.update_layout(
            height=500, font=dict(family="Inter"),
            xaxis_title=selected_feat, yaxis_title="Jumlah",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Korelasi ──
    with tab2:
        st.markdown("#### Heatmap Korelasi (Top 15 fitur terhadap dropout_risk)")
        corr = df[NUMERIC_FEATURES + ['dropout_risk']].corr()
        top_corr = corr['dropout_risk'].drop('dropout_risk').abs().sort_values(ascending=False).head(15)

        fig_corr = go.Figure(data=go.Heatmap(
            z=df[top_corr.index.tolist() + ['dropout_risk']].corr().values,
            x=top_corr.index.tolist() + ['dropout_risk'],
            y=top_corr.index.tolist() + ['dropout_risk'],
            colorscale='RdBu_r', zmid=0,
            text=np.round(df[top_corr.index.tolist() + ['dropout_risk']].corr().values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
        ))
        fig_corr.update_layout(
            height=600, font=dict(family="Inter", size=10),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### Top 15 Korelasi dengan Dropout Risk")
        corr_df = pd.DataFrame({
            'Fitur': top_corr.index,
            'Korelasi': corr['dropout_risk'][top_corr.index].values
        }).sort_values('Korelasi', key=abs, ascending=True)

        fig_bar = px.bar(
            corr_df, x='Korelasi', y='Fitur', orientation='h',
            color='Korelasi', color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
        )
        fig_bar.update_layout(height=450, font=dict(family="Inter"))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tab 3: Box Plot ──
    with tab3:
        st.markdown("#### Box Plot per Status Mahasiswa")
        box_feat = st.selectbox(
            "Pilih fitur:", NUMERIC_FEATURES,
            index=NUMERIC_FEATURES.index('Curricular units 2nd sem (approved)'),
            key='box_feat'
        )
        fig_box = px.box(
            df, x='Target', y=box_feat, color='Target',
            color_discrete_map=colors_map,
            points='outliers',
        )
        fig_box.update_layout(height=500, font=dict(family="Inter"), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Tab 4: Perbandingan ──
    with tab4:
        st.markdown("#### Rata-rata Fitur per Status")
        compare_feats = st.multiselect(
            "Pilih fitur untuk dibandingkan:",
            NUMERIC_FEATURES,
            default=[
                'Curricular units 1st sem (approved)',
                'Curricular units 2nd sem (approved)',
                'Curricular units 1st sem (grade)',
                'Curricular units 2nd sem (grade)',
                'Age at enrollment',
            ]
        )
        if compare_feats:
            grouped = df.groupby('Target')[compare_feats].mean().T
            grouped = grouped.reset_index()
            grouped.columns = ['Fitur'] + list(grouped.columns[1:])
            fig_comp = go.Figure()
            for status in ['Graduate', 'Dropout', 'Enrolled']:
                if status in grouped.columns:
                    fig_comp.add_trace(go.Bar(
                        name=status, x=grouped['Fitur'], y=grouped[status],
                        marker_color=colors_map.get(status, '#636e72'),
                    ))
            fig_comp.update_layout(
                barmode='group', height=500, font=dict(family="Inter"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE: PREDIKSI
# ═══════════════════════════════════════════════════════
elif page == "🤖 Prediksi":
    st.markdown(
        """
        <div class="main-header">
            <h1>🤖 Prediksi Risiko Dropout</h1>
            <p>Masukkan data mahasiswa untuk memprediksi risiko dropout menggunakan model Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = get_model()
    if model is None:
        st.error("⚠️ Model belum dilatih! Jalankan `python src/train_model.py` terlebih dahulu.")
        st.stop()

    df = get_raw_data()

    st.markdown("### 📝 Input Data Mahasiswa")
    st.markdown("Isi form di bawah ini dengan data mahasiswa yang ingin diprediksi.")

    # Organize into groups
    with st.expander("👤 Data Pribadi", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            marital = st.number_input("Status Pernikahan (1=Single, 2=Married, ...)", min_value=1, max_value=6, value=1, key='marital')
            gender = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki", key='gender')
            age = st.number_input("Usia Saat Mendaftar", min_value=17, max_value=70, value=19, key='age')
        with col2:
            nationality = st.number_input("Kewarganegaraan (kode)", min_value=1, max_value=109, value=1, key='nat')
            international = st.selectbox("Mahasiswa Internasional", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", key='intl')
            displaced = st.selectbox("Pengungsi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", key='disp')
        with col3:
            special_needs = st.selectbox("Kebutuhan Khusus", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", key='sn')
            scholarship = st.selectbox("Penerima Beasiswa", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", key='sch')

    with st.expander("🎓 Data Pendaftaran", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            app_mode = st.number_input("Mode Pendaftaran", min_value=1, max_value=57, value=1, key='app_mode')
            app_order = st.number_input("Urutan Pendaftaran", min_value=0, max_value=9, value=1, key='app_order')
            course = st.number_input("Kode Program Studi", min_value=33, max_value=9991, value=9238, key='course')
        with col2:
            prev_qual = st.number_input("Kualifikasi Sebelumnya", min_value=1, max_value=43, value=1, key='pq')
            prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=95.0, max_value=190.0, value=130.0, key='pqg')
            admission_grade = st.number_input("Nilai Masuk", min_value=95.0, max_value=190.0, value=127.0, key='ag')
        with col3:
            mother_qual = st.number_input("Pendidikan Ibu", min_value=1, max_value=44, value=19, key='mq')
            father_qual = st.number_input("Pendidikan Ayah", min_value=1, max_value=44, value=12, key='fq')
            mother_occ = st.number_input("Pekerjaan Ibu", min_value=0, max_value=194, value=5, key='mo')
            father_occ = st.number_input("Pekerjaan Ayah", min_value=0, max_value=195, value=9, key='fo')

    with st.expander("📚 Data Akademik Semester 1", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            cu1_credited = st.number_input("SKS Diakui Sem 1", min_value=0, max_value=20, value=0, key='cu1c')
            cu1_enrolled = st.number_input("SKS Diambil Sem 1", min_value=0, max_value=26, value=6, key='cu1e')
        with col2:
            cu1_evaluations = st.number_input("SKS Dievaluasi Sem 1", min_value=0, max_value=45, value=6, key='cu1ev')
            cu1_approved = st.number_input("SKS Lulus Sem 1", min_value=0, max_value=26, value=5, key='cu1a')
        with col3:
            cu1_grade = st.number_input("Rata-rata Nilai Sem 1", min_value=0.0, max_value=19.0, value=12.0, key='cu1g')
            cu1_without = st.number_input("SKS Tanpa Evaluasi Sem 1", min_value=0, max_value=12, value=0, key='cu1w')

    with st.expander("📚 Data Akademik Semester 2", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            cu2_credited = st.number_input("SKS Diakui Sem 2", min_value=0, max_value=19, value=0, key='cu2c')
            cu2_enrolled = st.number_input("SKS Diambil Sem 2", min_value=0, max_value=23, value=6, key='cu2e')
        with col2:
            cu2_evaluations = st.number_input("SKS Dievaluasi Sem 2", min_value=0, max_value=33, value=6, key='cu2ev')
            cu2_approved = st.number_input("SKS Lulus Sem 2", min_value=0, max_value=20, value=5, key='cu2a')
        with col3:
            cu2_grade = st.number_input("Rata-rata Nilai Sem 2", min_value=0.0, max_value=19.0, value=12.0, key='cu2g')
            cu2_without = st.number_input("SKS Tanpa Evaluasi Sem 2", min_value=0, max_value=12, value=0, key='cu2w')

    with st.expander("💰 Data Keuangan & Ekonomi", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            debtor = st.selectbox("Memiliki Hutang", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya", key='debt')
            tuition = st.selectbox("SPP Lunas", [0, 1], index=1, format_func=lambda x: "Tidak" if x == 0 else "Ya", key='tui')
        with col2:
            unemp_rate = st.number_input("Tingkat Pengangguran (%)", min_value=7.0, max_value=17.0, value=11.0, key='ur')
            inflation = st.number_input("Tingkat Inflasi (%)", min_value=-1.0, max_value=4.0, value=1.4, key='inf')
        with col3:
            gdp = st.number_input("GDP", min_value=-5.0, max_value=4.0, value=1.0, key='gdp')

    # Build input dataframe
    st.markdown("---")
    if st.button("🔮 Prediksi Sekarang", use_container_width=True, type="primary"):
        input_data = pd.DataFrame([{
            'Marital status': marital,
            'Application mode': app_mode,
            'Application order': app_order,
            'Course': course,
            'Daytime/evening attendance\t': 1,
            'Previous qualification': prev_qual,
            'Previous qualification (grade)': prev_qual_grade,
            'Nacionality': nationality,
            "Mother's qualification": mother_qual,
            "Father's qualification": father_qual,
            "Mother's occupation": mother_occ,
            "Father's occupation": father_occ,
            'Admission grade': admission_grade,
            'Displaced': displaced,
            'Educational special needs': special_needs,
            'Debtor': debtor,
            'Tuition fees up to date': tuition,
            'Gender': gender,
            'Scholarship holder': scholarship,
            'Age at enrollment': age,
            'International': international,
            'Curricular units 1st sem (credited)': cu1_credited,
            'Curricular units 1st sem (enrolled)': cu1_enrolled,
            'Curricular units 1st sem (evaluations)': cu1_evaluations,
            'Curricular units 1st sem (approved)': cu1_approved,
            'Curricular units 1st sem (grade)': cu1_grade,
            'Curricular units 1st sem (without evaluations)': cu1_without,
            'Curricular units 2nd sem (credited)': cu2_credited,
            'Curricular units 2nd sem (enrolled)': cu2_enrolled,
            'Curricular units 2nd sem (evaluations)': cu2_evaluations,
            'Curricular units 2nd sem (approved)': cu2_approved,
            'Curricular units 2nd sem (grade)': cu2_grade,
            'Curricular units 2nd sem (without evaluations)': cu2_without,
            'Unemployment rate': unemp_rate,
            'Inflation rate': inflation,
            'GDP': gdp,
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.markdown(
                f"""
                <div class="prediction-box pred-dropout">
                    <div class="pred-text">⚠️ BERISIKO DROPOUT</div>
                    <div class="pred-prob">Probabilitas Dropout: {probability[1]*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.warning("Mahasiswa ini memiliki risiko tinggi untuk dropout. Disarankan untuk melakukan intervensi dini.")
        else:
            st.markdown(
                f"""
                <div class="prediction-box pred-safe">
                    <div class="pred-text">✅ AMAN (Non-Dropout)</div>
                    <div class="pred-prob">Probabilitas Bertahan: {probability[0]*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success("Mahasiswa ini diprediksi aman dan akan menyelesaikan studinya.")

        # Probability gauge
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                title={'text': "Risiko Dropout", 'font': {'size': 18}},
                number={'suffix': '%', 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if prediction == 1 else "#00b894"},
                    'steps': [
                        {'range': [0, 30], 'color': '#d4efdf'},
                        {'range': [30, 60], 'color': '#fdebd0'},
                        {'range': [60, 100], 'color': '#fadbd8'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.8, 'value': 50
                    },
                }
            ))
            fig_gauge.update_layout(height=300, font=dict(family="Inter"))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            st.markdown("#### 📋 Detail Probabilitas")
            prob_df = pd.DataFrame({
                'Status': ['Non-Dropout', 'Dropout'],
                'Probabilitas': [probability[0], probability[1]]
            })
            fig_prob = px.bar(
                prob_df, x='Status', y='Probabilitas',
                color='Status',
                color_discrete_map={'Non-Dropout': '#00b894', 'Dropout': '#e74c3c'},
                text=prob_df['Probabilitas'].apply(lambda x: f'{x*100:.1f}%'),
            )
            fig_prob.update_layout(
                height=300, font=dict(family="Inter"),
                showlegend=False, yaxis_range=[0, 1],
            )
            fig_prob.update_traces(textposition='outside')
            st.plotly_chart(fig_prob, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE: EVALUASI MODEL
# ═══════════════════════════════════════════════════════
elif page == "📈 Evaluasi Model":
    st.markdown(
        """
        <div class="main-header">
            <h1>📈 Evaluasi Performa Model</h1>
            <p>Metrik performa dan analisis mendalam model yang telah dilatih</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = get_model()
    X_test, y_test = get_processed_data()

    if model is None or X_test is None:
        st.error("⚠️ Model atau data test belum tersedia. Jalankan training terlebih dahulu.")
        st.stop()

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, roc_curve
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Metrics cards
    st.markdown("### 📊 Metrik Performa")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, emoji in [
        (c1, 'Accuracy', acc, '🎯'),
        (c2, 'Precision', prec, '🔍'),
        (c3, 'Recall', rec, '📡'),
        (c4, 'F1-Score', f1, '⚖️'),
        (c5, 'ROC-AUC', auc, '📈'),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem">{emoji}</div>
                    <div class="metric-value">{value:.3f}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix & ROC Curve
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown("### 📋 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
            x=['Non-Dropout', 'Dropout'],
            y=['Non-Dropout', 'Dropout'],
            color_continuous_scale='Blues',
        )
        fig_cm.update_layout(
            height=400, font=dict(family="Inter", size=14),
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.markdown("### 📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'Model (AUC={auc:.3f})',
            line=dict(color='#667eea', width=3),
            fill='tozeroy', fillcolor='rgba(102,126,234,0.1)',
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random', line=dict(color='gray', dash='dash'),
        ))
        fig_roc.update_layout(
            height=400, font=dict(family="Inter"),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.1),
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Importance
    st.markdown("### 🌟 Feature Importance (Top 15)")
    try:
        rf_model = model.named_steps['model']
        preprocessor = model.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [f.replace('num__', '') for f in feature_names]
        importances = rf_model.feature_importances_

        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(15)

        fig_fi = px.bar(
            feat_imp, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Viridis',
        )
        fig_fi.update_layout(
            height=500, font=dict(family="Inter"),
            margin=dict(l=10, r=10, t=30, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.info(f"Feature importance tidak tersedia: {e}")

    # Prediction Distribution
    st.markdown("### 📊 Distribusi Probabilitas Prediksi")
    prob_df = pd.DataFrame({
        'Probabilitas Dropout': y_prob,
        'Aktual': ['Dropout' if y == 1 else 'Non-Dropout' for y in y_test]
    })
    fig_dist = px.histogram(
        prob_df, x='Probabilitas Dropout', color='Aktual',
        color_discrete_map={'Dropout': '#e74c3c', 'Non-Dropout': '#00b894'},
        barmode='overlay', nbins=50, opacity=0.7,
    )
    fig_dist.update_layout(
        height=400, font=dict(family="Inter"),
        xaxis_title="Probabilitas Dropout", yaxis_title="Jumlah",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
