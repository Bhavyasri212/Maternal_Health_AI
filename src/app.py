import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from preprocessing import DataPreprocessor
from model import build_multimodal_model
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
    model.load_weights('../output/best_maternal_model.keras')

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

    if 'Age' in active_features:
        st.markdown('<div class="vital-label">👤 Age (years)</div>', unsafe_allow_html=True)
        input_data['Age'] = st.slider("Age", 10, 60, key='Age', label_visibility="collapsed")

    if 'SystolicBP' in active_features:
        st.markdown('<div class="vital-label">💓 Systolic BP (mmHg)</div>', unsafe_allow_html=True)
        input_data['SystolicBP'] = st.slider("Systolic BP", 70.0, 160.0, key='SystolicBP', label_visibility="collapsed")

    if 'DiastolicBP' in active_features:
        st.markdown('<div class="vital-label">💓 Diastolic BP (mmHg)</div>', unsafe_allow_html=True)
        input_data['DiastolicBP'] = st.slider("Diastolic BP", 50.0, 100.0, key='DiastolicBP', label_visibility="collapsed")

with col2:
    st.markdown('<div style="font-weight: 700; color: var(--primary-teal); margin-bottom: 16px; font-size: 1.1rem;">Metabolic Markers</div>', unsafe_allow_html=True)

    if 'BS' in active_features:
        st.markdown('<div class="vital-label">🩸 Blood Sugar (mmol/L)</div>', unsafe_allow_html=True)
        input_data['BS'] = st.slider("Blood Sugar", 6.0, 19.0, key='BS', label_visibility="collapsed")

    if 'BodyTemp' in active_features:
        st.markdown('<div class="vital-label">🌡️ Body Temperature (°F)</div>', unsafe_allow_html=True)
        input_data['BodyTemp'] = st.slider("Body Temperature", 98.0, 103.0, key='BodyTemp', label_visibility="collapsed")

    if 'HeartRate' in active_features:
        st.markdown('<div class="vital-label">❤️ Heart Rate (bpm)</div>', unsafe_allow_html=True)
        input_data['HeartRate'] = st.slider("Heart Rate", 50.0, 120.0, key='HeartRate', label_visibility="collapsed")

for feat in active_features:
    if feat not in input_data:
        input_data[feat] = st.session_state[feat]

input_df = pd.DataFrame([input_data])[active_features]

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

if st.session_state.get('run_analysis', False):
    with st.spinner("🔬 ANALYZING PATIENT DATA..."):
        preds = model.predict([input_clin_scaled, input_ctg, input_act, input_img])
        risk_probs = preds[0][0]
        weight_pred = preds[1][0][0]

        winner = np.argmax(risk_probs)
        labels = ["Low Risk", "Mid Risk", "High Risk"]

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

        st.session_state.run_analysis = False


# --- NEW SECTION: EXPLAINABLE AI (SHAP) - OPTIMIZED FOR SPEED ---
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 AI REASONING (SHAP ANALYSIS)</div>', unsafe_allow_html=True)
        
        import shap
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder

        st.info("ℹ️ This graph shows exactly how much each clinical factor pushed the risk score up (Red) or down (Blue).")

        # 1. PREPARE BACKGROUND DATA
        df_numeric_shap = df_raw[active_features].copy()
        for col in df_numeric_shap.columns:
            if df_numeric_shap[col].dtype == 'object':
                le = LabelEncoder()
                df_numeric_shap[col] = le.fit_transform(df_numeric_shap[col].astype(str))

        # 2. Wrapper Function
        def model_wrapper(clin_data_batch):
            N = clin_data_batch.shape[0]
            ctg_batch = np.repeat(input_ctg, N, axis=0)
            act_batch = np.repeat(input_act, N, axis=0)
            img_batch = np.repeat(input_img, N, axis=0)
            return model.predict([clin_data_batch, ctg_batch, act_batch, img_batch], verbose=0)[0][:, winner]

        # 3. Create Explainer (SPEED HACK: Use only 5 samples)
        # 5 samples is enough for a visual demo and runs 10x faster
        background = scaler.transform(df_numeric_shap.sample(5, random_state=42)) 
        explainer = shap.KernelExplainer(model_wrapper, background)

        # 4. Calculate SHAP Values (Reduced nsamples for speed)
        with st.spinner("🧠 Analyzing Clinical Factors..."):
            # nsamples=100 limits the total model runs. Keeps it snappy.
            shap_values = explainer.shap_values(input_clin_scaled, nsamples=100)

        # 5. Plot
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
        
        plt.title(f"Why did the AI predict {labels[winner]}?", fontsize=14, fontweight='bold', color='#0d2818')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
<div class="dashboard-footer">
    <div class="footer-title">🏥 MATERNAL HEALTH AI PLATFORM</div>
    <div class="footer-text">Advanced Multimodal Risk Assessment System for Clinical Decision Support</div>
    <div class="footer-subtext">Powered by Deep Learning & Computer Vision | Clinical Use Only | Always consult healthcare professionals</div>
</div>
""", unsafe_allow_html=True)
