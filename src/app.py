import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from preprocessing import DataPreprocessor
from model import build_multimodal_model
from pathlib import Path
import os


st.set_page_config(
    page_title="Maternal Health AI Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
:root {
    --primary-teal: #20c997;
    --secondary-teal: #15aabf;
    --accent-mint: #a2f5a2;
    --light-bg: #f0fdf4;
    --card-bg: #ffffff;
    --text-primary: #0d2818;
    --text-secondary: #2d5a3d;
    --text-muted: #6b8e7a;
    --border-color: #c3fad8;
    --success-green: #16a34a;
    --warning-orange: #f59e0b;
    --danger-red: #dc2626;
}

* {
    margin: 0;
    padding: 0;
}

.stApp {
    background: linear-gradient(180deg, #f0fdf4 0%, #ecfdf5 100%);
}

.stApp p, .stApp span, .stApp div, .stMarkdown {
    color: var(--text-primary);
}

.dashboard-header {
    background: linear-gradient(135deg, rgba(32, 201, 151, 0.95) 0%, rgba(21, 170, 191, 0.95) 100%);
    padding: 48px 40px;
    border-radius: 20px;
    margin-bottom: 40px;
    box-shadow: 0 12px 40px rgba(32, 201, 151, 0.25);
    color: white;
    text-align: center;
}

.main-title {
    font-size: 3.2rem;
    font-weight: 900;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}

.subtitle {
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.95);
    margin-top: 12px;
    font-weight: 500;
}

.status-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.25);
    color: white;
    padding: 8px 24px;
    border-radius: 20px;
    font-weight: 700;
    margin-top: 16px;
    border: 1px solid rgba(255, 255, 255, 0.4);
}

.section-container {
    background: var(--card-bg);
    padding: 36px;
    border-radius: 16px;
    margin: 28px 0;
    border: 2px solid var(--border-color);
    box-shadow: 0 8px 24px rgba(32, 201, 151, 0.12);
}

.section-title {
    color: var(--primary-teal);
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    border-bottom: 3px solid var(--accent-mint);
    padding-bottom: 16px;
}

.preset-buttons {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}

.vital-input-group {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 20px;
}

.vital-card {
    background: linear-gradient(135deg, var(--light-bg) 0%, rgba(162, 245, 162, 0.15) 100%);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
}

.vital-label {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-secondary);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.vital-highlight {
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(32, 201, 151, 0.08);
    box-shadow: 0 0 0 2px rgba(32, 201, 151, 0.35);
}

.metric-card {
    background: var(--card-bg);
    padding: 28px;
    border-radius: 16px;
    border: 2px solid var(--border-color);
    box-shadow: 0 8px 24px rgba(32, 201, 151, 0.12);
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--primary-teal);
}

.card-title {
    color: var(--primary-teal);
    font-size: 1.3rem;
    font-weight: 800;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.risk-low {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border-left: 8px solid var(--success-green);
    padding: 28px;
    border-radius: 12px;
}

.risk-mid {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-left: 8px solid var(--warning-orange);
    padding: 28px;
    border-radius: 12px;
}

.risk-high {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-left: 8px solid var(--danger-red);
    padding: 28px;
    border-radius: 12px;
}

.risk-title {
    font-size: 1.4rem;
    font-weight: 900;
    margin-bottom: 8px;
}

.confidence-text {
    font-size: 1.1rem;
    font-weight: 600;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--primary-teal), var(--secondary-teal)) !important;
    color: white !important;
    border: none !important;
    padding: 16px 24px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 18px rgba(32, 201, 151, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(32, 201, 151, 0.4) !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--primary-teal), var(--accent-mint)) !important;
    border-radius: 8px;
}

.stAlert {
    background: var(--card-bg) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    border-left: 6px solid !important;
    border-top: none !important;
    border-right: none !important;
    border-bottom: none !important;
}

.stAlert-success {
    border-left-color: var(--success-green) !important;
    background: linear-gradient(135deg, #ecfdf5, #d1fae5) !important;
}

.stAlert-warning {
    border-left-color: var(--warning-orange) !important;
    background: linear-gradient(135deg, #fffbeb, #fef3c7) !important;
}

.stAlert-error {
    border-left-color: var(--danger-red) !important;
    background: linear-gradient(135deg, #fef2f2, #fee2e2) !important;
}

.stAlert-info {
    border-left-color: var(--primary-teal) !important;
    background: linear-gradient(135deg, #ecfdf5, #c3fad8) !important;
}

[data-testid="stFileUploader"] {
    background: var(--light-bg) !important;
    border: 2px dashed var(--primary-teal) !important;
    border-radius: 16px !important;
    padding: 28px !important;
}

[data-testid="stMetricValue"] {
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--primary-teal);
}

[data-testid="stMetricLabel"] {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-muted);
}

.results-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
    margin-bottom: 28px;
}

.clinical-finding {
    background: var(--light-bg);
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 12px;
    border-left: 4px solid var(--primary-teal);
}

.dashboard-footer {
    background: linear-gradient(135deg, rgba(32, 201, 151, 0.05) 0%, rgba(162, 245, 162, 0.05) 100%);
    border-top: 3px solid var(--primary-teal);
    border-radius: 16px;
    padding: 32px;
    margin-top: 48px;
    text-align: center;
}

.footer-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--primary-teal);
    margin-bottom: 8px;
}

.footer-text {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

.footer-subtext {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 12px;
}

@media (max-width: 1024px) {
    .results-grid {
        grid-template-columns: 1fr;
    }

    .vital-input-group {
        grid-template-columns: 1fr;
    }
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    prep = DataPreprocessor()
    df_raw = prep.loader.load_maternal_risk_data()

    X_fused, y_fused = prep.fuse_datasets()
    X_clin, X_ctg, X_act, X_img = X_fused

    model = build_multimodal_model(
        (X_clin.shape[1],),
        (X_ctg.shape[1], X_ctg.shape[2]),
        (X_act.shape[1], X_act.shape[2]),
        (128, 128, 1)
    )
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "output" / "best_maternal_model__v4.keras"

    model.load_weights(MODEL_PATH)

    all_possible = [
        'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate',
        'sleep_hours', 'phys_activity_level', 'stress_score',
        'education', 'income_category', 'urban_rural',
        'diet_quality', 'hemoglobin', 'iron_suppl', 'folic_suppl', 'diet_adherence'
    ]
    active_features = [c for c in all_possible if c in df_raw.columns]

    df_numeric = df_raw[active_features].copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            le = LabelEncoder()
            df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))

    df_numeric = df_numeric.apply(pd.to_numeric, errors="coerce")
    df_numeric = df_numeric.fillna(df_numeric.mean(numeric_only=True))

    scaler = StandardScaler()
    scaler.fit(df_numeric)

    subset = 200
    preds = model.predict([X_clin[:subset], X_ctg[:subset], X_act[:subset], X_img[:subset]], verbose=0)
    probs = preds[0]

    low_idx = np.argmax(probs[:, 0])
    high_idx = np.argmax(probs[:, 2])
    mid_idx = np.argmax(probs[:, 1])

    templates = {'Low': low_idx, 'Mid': mid_idx, 'High': high_idx}

    return model, scaler, active_features, df_raw, X_clin, X_ctg, X_act, X_img, templates


@st.cache_resource
def get_lime_explainer(df_raw, active_features, _scaler):
    df_numeric = df_raw[active_features].copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            le = LabelEncoder()
            df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))

    df_numeric = df_numeric.apply(pd.to_numeric, errors="coerce")
    df_numeric = df_numeric.fillna(df_numeric.mean(numeric_only=True))
    valid_df = df_numeric.dropna()

    if len(valid_df) >= 10:
        sample_df = valid_df.sample(10, random_state=42)
    elif len(valid_df) > 0:
        sample_df = valid_df
    else:
        sample_df = pd.DataFrame([np.zeros(len(active_features))], columns=active_features)

    background_data = _scaler.transform(sample_df)
    background_data = np.nan_to_num(background_data, nan=0.0, posinf=0.0, neginf=0.0)
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        background_data,
        feature_names=active_features,
        class_names=["Low Risk", "Mid Risk", "High Risk"],
        mode="classification",
        feature_selection="none",
        discretize_continuous=False
    )
    return explainer

try:
    model, scaler, active_features, df_raw, X_clin_ref, X_ctg_ref, X_act_ref, X_img_ref, templates = load_system()
except Exception as e:
    st.error(f"System Load Error: {e}")
    st.stop()

if 'SystolicBP' not in st.session_state:
    for feat in active_features:
        if feat not in st.session_state:
            if pd.api.types.is_numeric_dtype(df_raw[feat]):
                val = df_raw[feat].mean()
            else:
                val = 0.0
            st.session_state[feat] = float(val)

    st.session_state.scenario_mode = "Healthy / Normal Pregnancy"
    st.session_state.preset_label = "Custom"

if "explain_with_lime" not in st.session_state:
    st.session_state.explain_with_lime = False

st.markdown("""
<div class="dashboard-header">
    <div class="main-title">🏥 MATERNAL HEALTH AI PLATFORM</div>
    <div class="subtitle">Advanced Multimodal Risk Assessment & Fetal Growth Monitoring System</div>
    <div class="status-badge">Clinical Decision Support v1.0</div>
</div>
""", unsafe_allow_html=True)
def apply_preset(risk_type):
    idx = templates[risk_type]

    for feat in active_features:
        val = df_raw.iloc[idx][feat]
        if isinstance(val, str):
            val = 0.0
        st.session_state[feat] = float(val)

    if risk_type == 'Low':
        st.session_state.scenario_mode = "Healthy / Normal Pregnancy"
        st.session_state.preset_label = "Low Risk"
    elif risk_type == 'Mid':
        st.session_state.scenario_mode = "Healthy / Normal Pregnancy"
        st.session_state.preset_label = "Mid Risk"
    else:
        st.session_state.scenario_mode = "High Risk / Distress Scenario"
        st.session_state.preset_label = "High Risk"

    st.rerun()

st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 PATIENT PROFILE</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🟢 LOW RISK", use_container_width=True):
        apply_preset('Low')

with col2:
    if st.button("🟡 MID RISK", use_container_width=True):
        apply_preset('Mid')

with col3:
    if st.button("🔴 HIGH RISK", use_container_width=True):
        apply_preset('High')

with col4:
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.preset_label = "Custom"
        st.rerun()

st.markdown(f"""
<div style='background: linear-gradient(135deg, rgba(32, 201, 151, 0.1) 0%, rgba(162, 245, 162, 0.1) 100%); padding: 16px; border-radius: 10px; border-left: 4px solid var(--primary-teal); margin-top: 20px;'>
    <div style='font-weight: 700; color: var(--primary-teal); margin-bottom: 6px;'>Current Profile</div>
    <div style='font-size: 1.1rem; color: var(--text-secondary); font-weight: 600;'>{st.session_state.preset_label}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💓 PATIENT VITAL SIGNS</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

input_data = {}

with col1:
    st.markdown('<div style="font-weight: 700; color: var(--primary-teal); margin-bottom: 16px; font-size: 1.1rem;">Primary Vitals</div>', unsafe_allow_html=True)

    shap_feats = st.session_state.get("shap_top_features") or []
    lime_feats = st.session_state.get("lime_top_features") or []
    highlight_features = set(shap_feats) | set(lime_feats)

    def vital_label(name, text):
        if name in highlight_features:
            st.markdown(f'<div class="vital-label vital-highlight">{text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="vital-label">{text}</div>', unsafe_allow_html=True)

    if 'Age' in active_features:
        vital_label('Age', '👤 Age (years)')
        input_data['Age'] = st.slider("Age", 10, 60, key='Age', label_visibility="collapsed")

    if 'SystolicBP' in active_features:
        vital_label('SystolicBP', '💓 Systolic BP (mmHg)')
        input_data['SystolicBP'] = st.slider("Systolic BP", 70.0, 160.0, key='SystolicBP', label_visibility="collapsed")

    if 'DiastolicBP' in active_features:
        vital_label('DiastolicBP', '💓 Diastolic BP (mmHg)')
        input_data['DiastolicBP'] = st.slider("Diastolic BP", 50.0, 100.0, key='DiastolicBP', label_visibility="collapsed")

with col2:
    st.markdown('<div style="font-weight: 700; color: var(--primary-teal); margin-bottom: 16px; font-size: 1.1rem;">Metabolic Markers</div>', unsafe_allow_html=True)

    if 'BS' in active_features:
        vital_label('BS', '🩸 Blood Sugar (mmol/L)')
        input_data['BS'] = st.slider("Blood Sugar", 6.0, 19.0, key='BS', label_visibility="collapsed")

    if 'BodyTemp' in active_features:
        vital_label('BodyTemp', '🌡️ Body Temperature (°F)')
        input_data['BodyTemp'] = st.slider("Body Temperature", 98.0, 103.0, key='BodyTemp', label_visibility="collapsed")

    if 'HeartRate' in active_features:
        vital_label('HeartRate', '❤️ Heart Rate (bpm)')
        input_data['HeartRate'] = st.slider("Heart Rate", 50.0, 120.0, key='HeartRate', label_visibility="collapsed")

for feat in active_features:
    if feat not in input_data:
        input_data[feat] = st.session_state[feat]

input_df = pd.DataFrame([input_data])[active_features]
input_df = input_df.apply(pd.to_numeric, errors="coerce")
input_df = input_df.fillna(df_raw[active_features].mean(numeric_only=True))
input_df = input_df.fillna(0.0)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔬 CLINICAL CONTEXT</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div style="font-weight: 700; color: var(--text-secondary); margin-bottom: 12px;">Sensor Data Profile</div>', unsafe_allow_html=True)
    radio_idx = 0 if st.session_state.scenario_mode.startswith("Healthy") else 1
    scenario = st.radio(
        "Select clinical context",
        ("Healthy / Normal Pregnancy", "High Risk / Distress Scenario"),
        index=radio_idx,
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<div style="font-weight: 700; color: var(--text-secondary); margin-bottom: 12px;">Ultrasound Scan</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload ultrasound",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

st.markdown('</div>', unsafe_allow_html=True)

input_clin_scaled = scaler.transform(input_df)

if scenario == "Healthy / Normal Pregnancy":
    idx = templates['Low']
else:
    idx = templates['High']

input_ctg = X_ctg_ref[idx].reshape(1, 11, 1)
input_act = X_act_ref[idx].reshape(1, 50, 3)
input_img = X_img_ref[idx].reshape(1, 128, 128, 1)

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    img = np.array(img)
    img = cv2.resize(img, (128, 128)) / 255.0
    input_img = img.reshape(1, 128, 128, 1)

col1, col2, col3 = st.columns(3)

with col2:
    if st.button("🚀 RUN ANALYSIS", use_container_width=True):
        st.session_state.run_analysis = True

with col3:
    st.session_state.explain_with_lime = st.toggle(
        "Explain this decision with LIME",
        value=st.session_state.explain_with_lime
    )

if st.session_state.get('run_analysis', False):
    with st.spinner("🔬 ANALYZING PATIENT DATA..."):
        preds = model.predict([input_clin_scaled, input_ctg, input_act, input_img])
        risk_probs = preds[0][0]
        weight_pred = preds[1][0][0]

        winner = np.argmax(risk_probs)
        labels = ["Low Risk", "Mid Risk", "High Risk"]

        baseline_clin_scaled = np.zeros_like(input_clin_scaled)
        ctg_zero = np.zeros_like(input_ctg)
        act_zero = np.zeros_like(input_act)
        img_zero = np.zeros_like(input_img)

        preds_no_clin = model.predict([baseline_clin_scaled, input_ctg, input_act, input_img], verbose=0)
        risk_no_clin = preds_no_clin[0][0]

        preds_no_sensor = model.predict([input_clin_scaled, ctg_zero, act_zero, input_img], verbose=0)
        risk_no_sensor = preds_no_sensor[0][0]

        preds_no_img = model.predict([input_clin_scaled, input_ctg, input_act, img_zero], verbose=0)
        risk_no_img = preds_no_img[0][0]

        full_conf = float(risk_probs[winner])
        clin_delta = max(0.0, full_conf - float(risk_no_clin[winner]))
        sensor_delta = max(0.0, full_conf - float(risk_no_sensor[winner]))
        img_delta = max(0.0, full_conf - float(risk_no_img[winner]))

        total_delta = clin_delta + sensor_delta + img_delta
        if total_delta > 0:
            clin_share = clin_delta / total_delta
            sensor_share = sensor_delta / total_delta
            img_share = img_delta / total_delta
        else:
            clin_share = sensor_share = img_share = 1.0 / 3.0

        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 ANALYSIS RESULTS</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🎯 RISK ASSESSMENT</div>', unsafe_allow_html=True)

            if winner == 0:
                st.markdown(f'<div class="risk-low"><div class="risk-title">✅ {labels[winner].upper()}</div><div class="confidence-text">Confidence: <strong>{risk_probs[0]*100:.1f}%</strong></div></div>', unsafe_allow_html=True)
            elif winner == 1:
                st.markdown(f'<div class="risk-mid"><div class="risk-title">⚠️ {labels[winner].upper()}</div><div class="confidence-text">Confidence: <strong>{risk_probs[1]*100:.1f}%</strong></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-high"><div class="risk-title">🚨 {labels[winner].upper()}</div><div class="confidence-text">Confidence: <strong>{risk_probs[2]*100:.1f}%</strong></div></div>', unsafe_allow_html=True)

            st.progress(float(risk_probs[winner]))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">👶 FETAL GROWTH</div>', unsafe_allow_html=True)

            show_weight = False
            if uploaded_file is not None:
                show_weight = True
            elif st.session_state.preset_label != "Custom":
                show_weight = True

            if show_weight:
                st.metric("Estimated Fetal Weight", f"{weight_pred:.0f} g", delta=None)
                if weight_pred < 2500:
                    st.warning("⚠️ Low Birth Weight Detected")
                else:
                    st.success("✅ Weight Within Normal Range")
            else:
                st.info("📷 Upload ultrasound to enable weight estimation")
                st.metric("Estimated Fetal Weight", "—")

            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📈 RISK DISTRIBUTION</div>', unsafe_allow_html=True)

            risk_df = pd.DataFrame({
                'Risk Level': ['Low', 'Mid', 'High'],
                'Probability': [risk_probs[0], risk_probs[1], risk_probs[2]]
            })

            for i, row in risk_df.iterrows():
                st.markdown(f"**{row['Risk Level']} Risk**: {row['Probability']*100:.1f}%")
                st.progress(float(row['Probability']))

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 CLINICAL INTERPRETATION</div>', unsafe_allow_html=True)

        sys_bp = input_data['SystolicBP']
        dia_bp = input_data['DiastolicBP']
        bs = input_data['BS']
        hr = input_data['HeartRate']

        findings = []

        if sys_bp >= 140:
            findings.append(("🔴 CRITICAL: Systolic BP is critically elevated", f"({sys_bp:.0f} mmHg) - Immediate attention required", "error"))
        elif sys_bp > 120:
            findings.append(("🟡 WARNING: Systolic BP is elevated", f"({sys_bp:.0f} mmHg) - Monitor closely", "warning"))
        else:
            findings.append(("🟢 NORMAL: Systolic BP within range", f"({sys_bp:.0f} mmHg)", "success"))

        if bs >= 10:
            findings.append(("🔴 CRITICAL: Blood Sugar indicates diabetes risk", f"({bs:.1f} mmol/L) - Requires intervention", "error"))
        elif bs > 7.5:
            findings.append(("🟡 WARNING: Blood Sugar is borderline high", f"({bs:.1f} mmol/L) - Dietary review recommended", "warning"))
        else:
            findings.append(("🟢 NORMAL: Blood Sugar within range", f"({bs:.1f} mmol/L)", "success"))

        if hr > 100:
            findings.append(("🟡 WARNING: Heart Rate is elevated", f"({hr:.0f} bpm) - Consider stress factors", "warning"))
        elif hr < 60:
            findings.append(("🟡 NOTE: Heart Rate is low", f"({hr:.0f} bpm) - Monitor if symptomatic", "warning"))
        else:
            findings.append(("🟢 NORMAL: Heart Rate within range", f"({hr:.0f} bpm)", "success"))

        for finding, detail, level in findings:
            if level == "error":
                st.error(f"{finding}\n\n{detail}")
            elif level == "warning":
                st.warning(f"{finding}\n\n{detail}")
            else:
                st.success(f"{finding}\n\n{detail}")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 CLINICAL RECOMMENDATIONS</div>', unsafe_allow_html=True)

        if winner == 0:
            st.success("✅ **OVERALL ASSESSMENT** - Low Risk\n\nPatient vitals are stable and within normal range. Continue routine monitoring and standard prenatal care protocols.")
        elif winner == 1:
            st.warning("⚠️ **OVERALL ASSESSMENT** - Moderate Risk\n\nModerate risk indicators detected. Enhanced monitoring is recommended. Schedule clinical follow-up within 48 hours for reassessment.")
        else:
            st.error("🚨 **OVERALL ASSESSMENT** - High Risk\n\nSignificant risk factors identified requiring immediate attention. Consider urgent clinical intervention and specialist consultation.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔁 WHAT-IF RISK REDUCTION</div>', unsafe_allow_html=True)

        target_vitals = []
        for v in ["SystolicBP", "DiastolicBP", "BS", "HeartRate"]:
            if v in active_features:
                target_vitals.append(v)

        suggestions = []

        for feat in target_vitals:
            current_val = float(input_df.iloc[0][feat])
            series = df_raw[feat].dropna()
            if series.empty:
                continue

            direction = -1.0
            target_val = float(series.quantile(0.1))

            if direction < 0 and current_val <= target_val:
                continue

            grid = np.linspace(current_val, target_val, 6)
            grid = np.unique(np.round(grid, 3))

            cf_df = pd.concat([input_df.copy() for _ in grid], ignore_index=True)
            cf_df[feat] = grid
            cf_scaled = scaler.transform(cf_df[active_features])

            n = cf_scaled.shape[0]
            ctg_batch = np.repeat(input_ctg, n, axis=0)
            act_batch = np.repeat(input_act, n, axis=0)
            img_batch = np.repeat(input_img, n, axis=0)

            cf_preds = model.predict([cf_scaled, ctg_batch, act_batch, img_batch], verbose=0)
            cf_probs = cf_preds[0]

            best = None

            for i, val in enumerate(grid):
                p = cf_probs[i]
                cls = int(np.argmax(p))
                prob_w = float(p[winner])

                if cls < winner:
                    best = (val, cls, prob_w)
                    break

                if prob_w < float(risk_probs[winner]) - 0.05:
                    best = (val, cls, prob_w)
                    break

            if best is not None:
                new_val, new_cls, new_prob = best
                delta = current_val - new_val
                if delta > 0:
                    text = f"Lower {feat} by about {delta:.1f} units (to ~{new_val:.1f}) to reduce the predicted risk."
                    suggestions.append(text)

        if suggestions:
            for s_text in suggestions:
                st.markdown(f"- {s_text}")
        else:
            st.markdown("No small changes in primary vitals were found that clearly reduce risk within typical ranges.")

        st.markdown('</div>', unsafe_allow_html=True)


# --- NEW SECTION: EXPLAINABLE AI (SHAP) - OPTIMIZED FOR SPEED ---
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 AI REASONING</div>', unsafe_allow_html=True)

        reasoning_tab, shap_tab, lime_tab, conf_tab, scenario_tab = st.tabs(
            ["Clinical reasoning", "SHAP explanation", "LIME explanation", "Confidence analysis", "Scenario comparison"]
        )

        with reasoning_tab:
            risk_label = labels[winner]
            st.markdown(
                f"**Model focus:** The system currently leans towards **{risk_label}** based on the combination of vitals and sensor data."
            )
            st.markdown(
                "Review the clinical interpretation and what-if suggestions above to understand possible levers for risk reduction."
            )

        with shap_tab:
            import shap
            import matplotlib.pyplot as plt

            st.info("ℹ️ The AI system globally weighed how each clinical factor influenced this risk estimate.")

            df_numeric_shap = df_raw[active_features].copy()
            for col in df_numeric_shap.columns:
                if df_numeric_shap[col].dtype == 'object':
                    le = LabelEncoder()
                    df_numeric_shap[col] = le.fit_transform(df_numeric_shap[col].astype(str))

            df_numeric_shap = df_numeric_shap.apply(pd.to_numeric, errors="coerce")
            df_numeric_shap = df_numeric_shap.fillna(df_numeric_shap.mean(numeric_only=True))
            valid_shap = df_numeric_shap.dropna()

            def model_wrapper(clin_data_batch):
                N = clin_data_batch.shape[0]
                ctg_batch = np.repeat(input_ctg, N, axis=0)
                act_batch = np.repeat(input_act, N, axis=0)
                img_batch = np.repeat(input_img, N, axis=0)
                return model.predict([clin_data_batch, ctg_batch, act_batch, img_batch], verbose=0)[0][:, winner]

            if len(valid_shap) >= 5:
                background = scaler.transform(valid_shap.sample(5, random_state=42))
            elif len(valid_shap) > 0:
                background = scaler.transform(valid_shap)
            else:
                background = scaler.transform(df_numeric_shap.head(1))
            explainer = shap.KernelExplainer(model_wrapper, background)

            with st.spinner("🧠 Analyzing feature contributions (SHAP)..."):
                shap_values = explainer.shap_values(input_clin_scaled, nsamples=100)

            shap_abs = np.abs(shap_values[0])
            top_idx = np.argsort(-shap_abs)[:5]
            st.session_state["shap_top_features"] = [active_features[i] for i in top_idx]

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=input_df.iloc[0].values,
                    feature_names=active_features
                ),
                max_display=10,
                show=False
            )

            st.pyplot(fig, use_container_width=True)

        with lime_tab:
            st.info("ℹ️ The AI system locally focused on this patient’s vitals to justify the risk estimate.")
            explain_with_lime = st.session_state.get("explain_with_lime", False)

            if not explain_with_lime:
                st.warning("Enable the LIME toggle above to generate a localized explanation for this prediction.")
            else:
                lime_explainer = get_lime_explainer(df_raw, active_features, scaler)

                def lime_predict(clin_batch_scaled):
                    clin_batch_scaled = np.nan_to_num(clin_batch_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    n = clin_batch_scaled.shape[0]
                    ctg_batch = np.repeat(input_ctg, n, axis=0)
                    act_batch = np.repeat(input_act, n, axis=0)
                    img_batch = np.repeat(input_img, n, axis=0)

                    preds_local = model.predict(
                        [clin_batch_scaled, ctg_batch, act_batch, img_batch],
                        verbose=0
                    )

                    probs = np.asarray(preds_local[0], dtype=float)
                    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

                    row_sums = probs.sum(axis=1, keepdims=True)
                    row_sums = np.where(row_sums == 0, 1.0, row_sums)
                    probs = probs / row_sums

                    return probs

                with st.spinner("🧠 Generating local decision explanation (LIME)..."):
                    instance = np.nan_to_num(input_clin_scaled[0], nan=0.0, posinf=0.0, neginf=0.0)
                    explanation = lime_explainer.explain_instance(
                        instance,
                        lime_predict,
                        num_features=min(6, len(active_features)),
                        num_samples=750
                    )

                contributions = explanation.as_list()
                lime_df = pd.DataFrame(contributions, columns=["Feature", "Contribution"])
                lime_df["Direction"] = np.where(lime_df["Contribution"] >= 0, "Increases risk", "Decreases risk")
                lime_df["abs_contribution"] = lime_df["Contribution"].abs()

                lime_sorted = lime_df.sort_values("abs_contribution", ascending=False)
                lime_top_raw = lime_sorted["Feature"].tolist()[:5]
                lime_parsed = []
                for name in lime_top_raw:
                    base = str(name).split(" ")[0]
                    lime_parsed.append(base)
                st.session_state["lime_top_features"] = lime_parsed

                import altair as alt

                chart = alt.Chart(lime_df).mark_bar().encode(
                    x=alt.X("Contribution", title="Effect on predicted risk"),
                    y=alt.Y("Feature", sort=alt.SortField(field="abs_contribution", order="descending")),
                    color=alt.condition(
                        alt.datum.Contribution > 0,
                        alt.value("#dc2626"),
                        alt.value("#16a34a")
                    ),
                    tooltip=["Feature", "Contribution", "Direction"]
                )
                st.altair_chart(chart, use_container_width=True)

                positive = lime_df[lime_df["Contribution"] > 0]["Feature"].tolist()
                negative = lime_df[lime_df["Contribution"] < 0]["Feature"].tolist()

                pos_text = ", ".join(positive[:3])
                neg_text = ", ".join(negative[:3])

                if positive and negative:
                    explanation_text = f"The AI system locally focused on {pos_text}, which increased the predicted risk, while {neg_text} helped to reduce it."
                elif positive:
                    explanation_text = f"The AI system locally focused on {pos_text}, which increased the predicted risk for this patient."
                elif negative:
                    explanation_text = f"The AI system locally focused on {neg_text}, which helped to reduce the predicted risk for this patient."
                else:
                    explanation_text = "The AI system did not identify any single vital sign with a dominant influence for this prediction."

                st.markdown(explanation_text)

        with conf_tab:
            clin_pct = int(clin_share * 100)
            sensor_pct = int(sensor_share * 100)
            img_pct = int(img_share * 100)

            st.markdown(
                f"<div style='display:flex; gap:16px; margin-bottom:16px;'>"
                f"<div class='clinical-finding'><strong>Clinical data</strong><br/>{clin_pct}% of risk confidence</div>"
                f"<div class='clinical-finding'><strong>Sensors (CTG + activity)</strong><br/>{sensor_pct}% of risk confidence</div>"
                f"<div class='clinical-finding'><strong>Ultrasound image</strong><br/>{img_pct}% of risk confidence</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            shap_feats = st.session_state.get("shap_top_features")
            lime_feats = st.session_state.get("lime_top_features")

            if shap_feats and lime_feats:
                shap_set = set(shap_feats)
                lime_set = set(lime_feats)
                inter = len(shap_set & lime_set)
                union = len(shap_set | lime_set)
                agreement = inter / union if union else 0.0

                if agreement >= 0.6:
                    level = "High agreement"
                    color = "#16a34a"
                    desc = "SHAP and LIME highlight largely the same key drivers."
                elif agreement >= 0.3:
                    level = "Medium agreement"
                    color = "#f59e0b"
                    desc = "Explanations overlap partially. Interpret with some caution."
                else:
                    level = "Low agreement"
                    color = "#dc2626"
                    desc = "SHAP and LIME disagree. Treat this decision as less certain."

                score_pct = int(agreement * 100)
                st.markdown(
                    f"<div style='margin-top:16px; padding:12px 16px; border-radius:10px; "
                    f"border:2px solid {color}; background: rgba(255,255,255,0.85);'>"
                    f"<strong>Explanation Consistency Meter</strong><br/>"
                    f"<span style='color:{color}; font-weight:700;'>{level}</span> "
                    f"(overlap {score_pct}%)<br/>"
                    f"<span style='color:var(--text-secondary); font-size:0.9rem;'>{desc}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

        with scenario_tab:
            st.info("Compare the current patient profile against low-risk and high-risk reference profiles.")

            scenarios = []
            label_map = {
                "current": "Current patient",
                "low": "Low-risk template",
                "high": "High-risk template"
            }
            vital_keys = ["SystolicBP", "DiastolicBP", "BS", "HeartRate"]

            current_row = {}
            for feat in vital_keys:
                if feat in input_data:
                    current_row[feat] = float(input_data[feat])
            if current_row:
                current_row["Scenario"] = label_map["current"]
                scenarios.append(current_row)

            for key, template_key in [("low", "Low"), ("high", "High")]:
                idx_template = templates.get(template_key)
                if idx_template is None:
                    continue
                base_row = df_raw.iloc[idx_template]
                row = {}
                for feat in vital_keys:
                    if feat in active_features and feat in base_row.index:
                        val = base_row[feat]
                        if isinstance(val, str):
                            continue
                        try:
                            row[feat] = float(val)
                        except Exception:
                            continue
                if row:
                    row["Scenario"] = label_map[key]
                    scenarios.append(row)

            if scenarios:
                comp_df = pd.DataFrame(scenarios)
                value_cols = [c for c in vital_keys if c in comp_df.columns]

                if value_cols:
                    records = []
                    for _, r in comp_df.iterrows():
                        for feat in value_cols:
                            records.append(
                                {
                                    "Scenario": r["Scenario"],
                                    "Metric": feat,
                                    "Value": float(r[feat])
                                }
                            )
                    long_df = pd.DataFrame(records)

                    import altair as alt

                    chart = alt.Chart(long_df).mark_bar().encode(
                        x=alt.X("Scenario:N", title="Scenario"),
                        y=alt.Y("Value:Q", title="Value"),
                        color=alt.Color("Scenario:N", legend=None),
                        column=alt.Column("Metric:N", title=None)
                    ).properties(height=220)

                    st.altair_chart(chart, use_container_width=True)

                    lines = []
                    current_name = label_map["current"]
                    low_name = label_map["low"]
                    high_name = label_map["high"]

                    for feat in value_cols:
                        if current_name not in comp_df["Scenario"].values:
                            continue
                        current_val = float(
                            comp_df.loc[comp_df["Scenario"] == current_name, feat].iloc[0]
                        )

                        diffs = []
                        if low_name in comp_df["Scenario"].values:
                            low_val = float(
                                comp_df.loc[comp_df["Scenario"] == low_name, feat].iloc[0]
                            )
                            diffs.append(("low", abs(current_val - low_val)))
                        if high_name in comp_df["Scenario"].values:
                            high_val = float(
                                comp_df.loc[comp_df["Scenario"] == high_name, feat].iloc[0]
                            )
                            diffs.append(("high", abs(current_val - high_val)))

                        if not diffs:
                            continue

                        nearest = min(diffs, key=lambda x: x[1])[0]
                        if nearest == "high":
                            lines.append(f"{feat} is closer to the high-risk template than the low-risk template.")
                        elif nearest == "low":
                            lines.append(f"{feat} is closer to the low-risk template than the high-risk template.")

                    if lines:
                        st.markdown("Key differences in primary vitals:")
                        for t in lines:
                            st.markdown(f"- {t}")
            else:
                st.info("Comparison not available because required vital signs are missing.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📄 CLINICAL SUMMARY PDF</div>', unsafe_allow_html=True)

        shap_feats = st.session_state.get("shap_top_features") or []
        lime_feats = st.session_state.get("lime_top_features") or []

        profile_text = f"Profile: {st.session_state.preset_label} | Context: {scenario}"
        risk_text = f"Risk class: {labels[winner]} ({risk_probs[winner]*100:.1f}% confidence)"

        dist_lines = []
        for i, lbl in enumerate(labels):
            dist_lines.append(f"{lbl}: {risk_probs[i]*100:.1f}%")

        vitals_lines = []
        for feat in ["SystolicBP", "DiastolicBP", "BS", "HeartRate"]:
            if feat in input_data:
                vitals_lines.append(f"{feat}: {input_data[feat]}")

        findings_lines = []
        for finding, detail, level in findings:
            findings_lines.append(f"{finding} {detail}")

        if winner == 0:
            overall_text = "OVERALL ASSESSMENT: Low Risk. Patient vitals are stable and within normal range. Continue routine monitoring and standard prenatal care."
        elif winner == 1:
            overall_text = "OVERALL ASSESSMENT: Moderate Risk. Moderate risk indicators detected. Enhanced monitoring and clinical follow-up within 48 hours is recommended."
        else:
            overall_text = "OVERALL ASSESSMENT: High Risk. Significant risk factors identified requiring urgent clinical review and possible intervention."

        if shap_feats:
            shap_text = "Top SHAP drivers: " + ", ".join(shap_feats)
        else:
            shap_text = "Top SHAP drivers: not available in this session."

        if lime_feats:
            lime_text = "Top LIME drivers: " + ", ".join(lime_feats)
        else:
            lime_text = "Top LIME drivers: not available in this session."

        if suggestions:
            what_if_text = ["What-if suggestions:"]
            what_if_text.extend(f"- {s}" for s in suggestions)
        else:
            what_if_text = ["No clear low-effort risk reduction suggestions for primary vitals."]

        import matplotlib.pyplot as plt
        import io

        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.95
        line_h = 0.03

        fig.text(0.1, y, "Maternal Health AI – Clinical Summary", fontsize=16, weight="bold")
        y -= line_h * 2

        fig.text(0.1, y, profile_text, fontsize=11)
        y -= line_h
        fig.text(0.1, y, risk_text, fontsize=11)
        y -= line_h
        fig.text(0.1, y, overall_text, fontsize=10)
        y -= line_h * 2

        fig.text(0.1, y, "Risk distribution:", fontsize=12, weight="bold")
        y -= line_h
        for line in dist_lines:
            fig.text(0.12, y, line, fontsize=10)
            y -= line_h

        y -= line_h / 2
        fig.text(0.1, y, "Primary vitals:", fontsize=12, weight="bold")
        y -= line_h
        for line in vitals_lines:
            fig.text(0.12, y, line, fontsize=10)
            y -= line_h

        y -= line_h / 2
        fig.text(0.1, y, "Key clinical findings:", fontsize=12, weight="bold")
        y -= line_h
        for line in findings_lines:
            if y < 0.12:
                break
            fig.text(0.12, y, line, fontsize=9)
            y -= line_h

        if y > 0.18:
            y -= line_h / 2
            fig.text(0.1, y, "AI explanations:", fontsize=12, weight="bold")
            y -= line_h
            fig.text(0.12, y, shap_text, fontsize=9)
            y -= line_h
            fig.text(0.12, y, lime_text, fontsize=9)
            y -= line_h

        if y > 0.18:
            y -= line_h / 2
            fig.text(0.1, y, "What-if analysis:", fontsize=12, weight="bold")
            y -= line_h
            for line in what_if_text:
                if y < 0.07:
                    break
                fig.text(0.12, y, line, fontsize=9)
                y -= line_h

        buf = io.BytesIO()
        fig.savefig(buf, format="pdf")
        plt.close(fig)
        buf.seek(0)
        pdf_bytes = buf.getvalue()

        st.download_button(
            "📄 Download Clinical Summary (PDF)",
            data=pdf_bytes,
            file_name="maternal_clinical_summary.pdf",
            mime="application/pdf"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.run_analysis = False
        st.success("Analysis complete. Review AI reasoning and clinical interpretation above.")


st.markdown("""
<div class="dashboard-footer">
    <div class="footer-title">🏥 MATERNAL HEALTH AI PLATFORM</div>
    <div class="footer-text">Advanced Multimodal Risk Assessment System for Clinical Decision Support</div>
    <div class="footer-subtext">Powered by Deep Learning & Computer Vision | Clinical Use Only | Always consult healthcare professionals</div>
</div>
""", unsafe_allow_html=True)