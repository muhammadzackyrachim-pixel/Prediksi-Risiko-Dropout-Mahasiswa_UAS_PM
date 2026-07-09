import streamlit as st
import pandas as pd
import joblib
import os

st.title("🎯 Predict Output")
st.write("Masukkan data mahasiswa untuk memprediksi risiko dropout.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Input Data Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    marital_status = st.number_input("Marital status", min_value=0, value=1)
    application_mode = st.number_input("Application mode", min_value=0, value=1)
    application_order = st.number_input("Application order", min_value=0, value=1)
    course = st.number_input("Course", min_value=0, value=171)
    attendance = st.number_input("Daytime/evening attendance", min_value=0, value=1)
    prev_qualification = st.number_input("Previous qualification", min_value=0, value=1)
    prev_qualification_grade = st.number_input("Previous qualification (grade)", min_value=0.0, value=120.0)
    nationality = st.number_input("Nacionality", min_value=0, value=1)
    mothers_qualification = st.number_input("Mother's qualification", min_value=0, value=1)
    fathers_qualification = st.number_input("Father's qualification", min_value=0, value=1)
    mothers_occupation = st.number_input("Mother's occupation", min_value=0, value=1)
    fathers_occupation = st.number_input("Father's occupation", min_value=0, value=1)
    admission_grade = st.number_input("Admission grade", min_value=0.0, value=120.0)
    displaced = st.number_input("Displaced", min_value=0, value=0)
    special_needs = st.number_input("Educational special needs", min_value=0, value=0)
    debtor = st.number_input("Debtor", min_value=0, max_value=1, value=0)
    tuition_fees = st.number_input("Tuition fees up to date", min_value=0, max_value=1, value=1)
    gender = st.number_input("Gender", min_value=0, max_value=1, value=0)

with col2:
    scholarship_holder = st.number_input("Scholarship holder", min_value=0, max_value=1, value=0)
    age_at_enrollment = st.number_input("Age at enrollment", min_value=15, value=20)
    international = st.number_input("International", min_value=0, max_value=1, value=0)
    cu1_credited = st.number_input("Curricular units 1st sem (credited)", min_value=0, value=0)
    cu1_enrolled = st.number_input("Curricular units 1st sem (enrolled)", min_value=0, value=6)
    cu1_evaluations = st.number_input("Curricular units 1st sem (evaluations)", min_value=0, value=6)
    cu1_approved = st.number_input("Curricular units 1st sem (approved)", min_value=0, value=5)
    cu1_grade = st.number_input("Curricular units 1st sem (grade)", min_value=0.0, value=12.0)
    cu1_without_eval = st.number_input("Curricular units 1st sem (without evaluations)", min_value=0, value=0)
    cu2_credited = st.number_input("Curricular units 2nd sem (credited)", min_value=0, value=0)
    cu2_enrolled = st.number_input("Curricular units 2nd sem (enrolled)", min_value=0, value=6)
    cu2_evaluations = st.number_input("Curricular units 2nd sem (evaluations)", min_value=0, value=6)
    cu2_approved = st.number_input("Curricular units 2nd sem (approved)", min_value=0, value=5)
    cu2_grade = st.number_input("Curricular units 2nd sem (grade)", min_value=0.0, value=12.0)
    cu2_without_eval = st.number_input("Curricular units 2nd sem (without evaluations)", min_value=0, value=0)
    unemployment_rate = st.number_input("Unemployment rate", value=10.8)
    inflation_rate = st.number_input("Inflation rate", value=1.4)
    gdp = st.number_input("GDP", value=1.74)

if st.button("Prediksi"):
    input_df = pd.DataFrame([{
        'Marital status': marital_status,
        'Application mode': application_mode,
        'Application order': application_order,
        'Course': course,
        'Daytime/evening attendance\t': attendance,
        'Previous qualification': prev_qualification,
        'Previous qualification (grade)': prev_qualification_grade,
        'Nacionality': nationality,
        "Mother's qualification": mothers_qualification,
        "Father's qualification": fathers_qualification,
        "Mother's occupation": mothers_occupation,
        "Father's occupation": fathers_occupation,
        'Admission grade': admission_grade,
        'Displaced': displaced,
        'Educational special needs': special_needs,
        'Debtor': debtor,
        'Tuition fees up to date': tuition_fees,
        'Gender': gender,
        'Scholarship holder': scholarship_holder,
        'Age at enrollment': age_at_enrollment,
        'International': international,
        'Curricular units 1st sem (credited)': cu1_credited,
        'Curricular units 1st sem (enrolled)': cu1_enrolled,
        'Curricular units 1st sem (evaluations)': cu1_evaluations,
        'Curricular units 1st sem (approved)': cu1_approved,
        'Curricular units 1st sem (grade)': cu1_grade,
        'Curricular units 1st sem (without evaluations)': cu1_without_eval,
        'Curricular units 2nd sem (credited)': cu2_credited,
        'Curricular units 2nd sem (enrolled)': cu2_enrolled,
        'Curricular units 2nd sem (evaluations)': cu2_evaluations,
        'Curricular units 2nd sem (approved)': cu2_approved,
        'Curricular units 2nd sem (grade)': cu2_grade,
        'Curricular units 2nd sem (without evaluations)': cu2_without_eval,
        'Unemployment rate': unemployment_rate,
        'Inflation rate': inflation_rate,
        'GDP': gdp
    }])

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("Hasil Prediksi: Mahasiswa berisiko **Dropout**")
    else:
        st.success("Hasil Prediksi: Mahasiswa **Tidak Berisiko Dropout**")