import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config without sidebar
st.set_page_config(
    page_title="Heart Health AI Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (CSS for clean spacing & spacious forms)
st.markdown("""
    <style>
    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main Layout Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }
    
    .main-header {
        font-size: 2.4rem;
        color: #E63946;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Input Field Padding */
    .stNumberInput, .stSelectbox {
        margin-bottom: 0.8rem;
    }

    /* Submit Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #E63946;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.8rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0px 4px 10px rgba(230, 57, 70, 0.3);
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        background-color: #D62828;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<div class='main-header'>❤️ Cardiac Health Assessment</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Enter patient clinical data to calculate heart disease risk probability.</div>", unsafe_allow_html=True)

# Load Model & Scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")       
    scaler = joblib.load("scaler.pkl")     
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler files ('model.pkl', 'scaler.pkl'). Details: {e}")

# Patient Input Form
with st.form("patient_form"):
    
    # --- SECTION 1: DEMOGRAPHICS & VITALS ---
    st.markdown("### 1️⃣ Patient Vitals & Demographics")
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Gender", options=["Male", "Female"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)

    with col2:
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No (0)", "Yes (1)"])

    st.markdown("---")

    # --- SECTION 2: CARDIAC & EXERCISE METRICS ---
    st.markdown("### 2️⃣ Cardiac & Exercise Measurements")
    col3, col4 = st.columns(2, gap="large")

    with col3:
        max_hr = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=230, value=150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    with col4:
        chest_pain = st.selectbox("Chest Pain Type", options=[
            "Asymptomatic (ASY)", 
            "Atypical Angina (ATA)", 
            "Non-Anginal Pain (NAP)", 
            "Typical Angina (TA)"
        ])
        resting_ecg = st.selectbox("Resting ECG Results", options=[
            "LV Hypertrophy (LVH)", 
            "Normal", 
            "ST-T Wave Abnormality (ST)"
        ])
        st_slope = st.selectbox("ST Slope", options=[
            "Down-sloping (Down)", 
            "Flat", 
            "Up-sloping (Up)"
        ])

    st.markdown("---")
    submit_button = st.form_submit_button("🔍 Run Prediction")

# Processing Prediction
if submit_button:
    # 1. Map Categorical Inputs
    sex_val = 1 if sex == "Male" else 0
    fasting_bs_val = 1 if "Yes" in fasting_bs else 0
    exercise_angina_val = 1 if exercise_angina == "Yes" else 0

    # One-Hot Encoding for ChestPainType (Baseline: ASY)
    cp_ata = 1 if chest_pain == "Atypical Angina (ATA)" else 0
    cp_nap = 1 if chest_pain == "Non-Anginal Pain (NAP)" else 0
    cp_ta  = 1 if chest_pain == "Typical Angina (TA)" else 0

    # One-Hot Encoding for RestingECG (Baseline: LVH)
    ecg_normal = 1 if resting_ecg == "Normal" else 0
    ecg_st     = 1 if "ST-T" in resting_ecg else 0

    # One-Hot Encoding for ST_Slope (Baseline: Down)
    slope_flat = 1 if "Flat" in st_slope else 0
    slope_up   = 1 if "Up" in st_slope else 0

    # 2. Scale continuous features using scaler.pkl
    continuous_df = pd.DataFrame([[age, resting_bp, cholesterol, max_hr]], 
                                 columns=['Age', 'RestingBP', 'Cholesterol', 'MaxHR'])
    scaled_continuous = scaler.transform(continuous_df)

    scaled_age = scaled_continuous[0][0]
    scaled_rbp = scaled_continuous[0][1]
    scaled_chol = scaled_continuous[0][2]
    scaled_max_hr = scaled_continuous[0][3]

    # 3. Create feature array matching model.pkl
    input_features = pd.DataFrame([[
        scaled_age,
        sex_val,
        scaled_rbp,
        scaled_chol,
        fasting_bs_val,
        scaled_max_hr,
        exercise_angina_val,
        oldpeak,
        cp_ata,
        cp_nap,
        cp_ta,
        ecg_normal,
        ecg_st,
        slope_flat,
        slope_up
    ]], columns=[
        'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 
        'ExerciseAngina', 'Oldpeak', 'ChestPainType_ATA', 'ChestPainType_NAP', 
        'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 
        'ST_Slope_Flat', 'ST_Slope_Up'
    ])

    # 4. Predict
    prediction = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]

    # Results Display
    st.markdown("## 📊 Result")
    
    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **High Risk of Heart Disease**\n\nPatient records indicate a higher probability of cardiac complications.")
        else:
            st.success("✅ **Low Risk of Heart Disease**\n\nPatient parameters fall within standard non-risk ranges.")

    with res_col2:
        risk_percentage = probabilities[1] * 100
        st.metric(label="Calculated Risk Probability", value=f"{risk_percentage:.1f}%")
        st.progress(int(risk_percentage))