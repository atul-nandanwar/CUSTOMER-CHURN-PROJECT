import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="CHURN-MATRIX // Cyber Retention Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Ultra-Futuristic Cybernetic Dark Theme CSS & Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@500;600;700&display=swap');

    /* Global Cyberpunk Base */
    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1224 0%, #05070e 70%, #020307 100%);
        color: #e2e8f0;
    }

    /* Holographic Glow Header */
    .cyber-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #00f0ff 0%, #7000ff 50%, #ff007b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 240, 255, 0.4);
        margin-bottom: 2px;
    }

    .cyber-badge {
        font-family: 'Orbitron', monospace;
        font-size: 0.75rem;
        padding: 4px 10px;
        background: rgba(0, 240, 255, 0.1);
        border: 1px solid #00f0ff;
        color: #00f0ff;
        border-radius: 4px;
        letter-spacing: 1px;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
        display: inline-block;
        margin-bottom: 20px;
    }

    /* Cybernetic Glass Panels */
    .cyber-panel {
        background: rgba(13, 18, 36, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 240, 255, 0.18);
        border-radius: 12px;
        padding: 22px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .cyber-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: linear-gradient(180deg, #00f0ff, transparent);
    }

    .cyber-panel:hover {
        border-color: rgba(0, 240, 255, 0.6);
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.25);
        transform: translateY(-2px);
    }

    .panel-header {
        font-family: 'Orbitron', monospace;
        font-size: 0.95rem;
        color: #00f0ff;
        letter-spacing: 1.5px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Futuristic Animated Cyber Button */
    div.stButton > button:first-child {
        font-family: 'Orbitron', monospace;
        background: linear-gradient(90deg, #7000ff 0%, #00f0ff 100%);
        color: #ffffff;
        font-weight: 800;
        font-size: 1.05rem;
        letter-spacing: 2px;
        border: 1px solid #00f0ff;
        border-radius: 8px;
        padding: 14px 28px;
        text-transform: uppercase;
        box-shadow: 0 0 20px rgba(112, 0, 255, 0.5);
        transition: all 0.3s ease-in-out;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #00f0ff 0%, #ff007b 100%);
        box-shadow: 0 0 35px rgba(0, 240, 255, 0.8);
        transform: scale(1.01);
        color: #ffffff;
    }

    /* Risk Status Cards */
    .danger-terminal {
        background: rgba(255, 0, 85, 0.12);
        border: 1px solid #ff0055;
        box-shadow: 0 0 25px rgba(255, 0, 85, 0.3);
        border-radius: 10px;
        padding: 18px;
        color: #ff4d79;
        font-family: 'Orbitron', monospace;
    }

    .safe-terminal {
        background: rgba(0, 255, 170, 0.12);
        border: 1px solid #00ffaa;
        box-shadow: 0 0 25px rgba(0, 255, 170, 0.3);
        border-radius: 10px;
        padding: 18px;
        color: #00ffaa;
        font-family: 'Orbitron', monospace;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Loader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

bundle = load_bundle()

# Gauge Chart Function
def create_gauge(probability):
    gauge_color = "#00ffaa" if probability < 0.35 else "#f59e0b" if probability < 0.65 else "#ff0055"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 42, 'family': "Orbitron", 'color': "#ffffff"}},
        title={'text': "NEURAL RISK ASSESSMENT", 'font': {'size': 14, 'family': "Orbitron", 'color': "#94a3b8"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': "rgba(15, 23, 42, 0.8)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(0, 255, 170, 0.1)'},
                {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [65, 100], 'color': 'rgba(255, 0, 85, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=260,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig

# 4. App Header
st.markdown('<div class="cyber-title">CHURN-MATRIX // RETENTION OS</div>', unsafe_allow_html=True)
st.markdown('<div class="cyber-badge">SYSTEM PROTOCOL: ONLINE • NEURAL ENGINE v3.1</div>', unsafe_allow_html=True)

if bundle is None:
    st.error("⚠️ CRITICAL: Model artifacts missing. Run `python notebooks/model_training.py`.")
else:
    model = bundle["model"]
    model_name = bundle["model_name"]
    label_encoders = bundle["label_encoders"]
    feature_names = bundle["feature_names"]

    # Navigation Tabs
    tab1, tab2 = st.tabs(["⚡ REAL-TIME TELEMETRY (SINGLE)", "📂 BULK MATRIX INFERENCE (CSV)"])

    # TAB 1: Single Diagnostic
    with tab1:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="cyber-panel"><div class="panel-header">◈ IDENTITY & DEMOGRAPHICS</div>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen Status", ["No", "Yes"])
            partner = st.selectbox("Partner Ecosystem", ["No", "Yes"])
            dependents = st.selectbox("Family Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure Duration (Months)", 0, 72, 12)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="cyber-panel"><div class="panel-header">◈ CONNECTIVITY NODES</div>', unsafe_allow_html=True)
            phone_service = st.selectbox("Voice Line", ["Yes", "No"])
            multiple_lines = st.selectbox("Multi-Line Routing", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Core Protocol", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Cyber Shield", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Cloud Vault", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Hardware Guard", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Priority Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Stream Feed TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Stream Cine Feed", ["No", "Yes", "No internet service"])
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="cyber-panel"><div class="panel-header">◈ FISCAL PROTOCOLS</div>', unsafe_allow_html=True)
            contract = st.selectbox("SLA Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Digital Ledger (Paperless)", ["Yes", "No"])
            payment_method = st.selectbox("Gateway Route", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Fee ($)", min_value=15.0, max_value=200.0, value=65.0, step=0.5)
            total_charges = st.number_input("Lifetime Value ($)", min_value=0.0, max_value=12000.0, value=round(float(tenure * monthly_charges), 2), step=10.0)
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        compute_btn = st.button("EXECUTE NEURAL RISK AUDIT ⚡", use_container_width=True)

        if compute_btn:
            input_dict = {
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }

            input_df = pd.DataFrame([input_dict])[feature_names]

            for col, le in label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except ValueError:
                        input_df[col] = 0

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("---")
            rc1, rc2 = st.columns([1.2, 1])

            with rc1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="danger-terminal">
                        <h3>⚠️ HIGH ATTRITION PROBABILITY DETECTED</h3>
                        <p>Telemetry flags severe disconnect risks. Intercept protocol required immediately.</p>
                        <hr style="border-color: rgba(255,0,85,0.3)">
                        <b>⚡ AI Retention Directive:</b> Offer 1-year contract lock discount (15% off) + activate Free Cyber Shield Security.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-terminal">
                        <h3>✅ CLIENT RETENTION SECURE</h3>
                        <p>Customer indicates strong loyalty vectors and minimal attrition patterns.</p>
                        <hr style="border-color: rgba(0,255,170,0.3)">
                        <b>⚡ AI Growth Directive:</b> Target for premium add-ons (Multi-line / Cloud Vault expansions).
                    </div>
                    """, unsafe_allow_html=True)

            with rc2:
                st.plotly_chart(create_gauge(probability), use_container_width=True)

    # TAB 2: Batch CSV Prediction
    with tab2:
        st.markdown('<div class="cyber-panel"><div class="panel-header">◈ BULK CSV TELEMETRY INGESTION</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file for mass churn scanning", type=["csv"])
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(batch_df)} customer rows.")
            
            if st.button("RUN BULK INFERENCE", type="primary"):
                processed_df = batch_df.copy()
                
                if "customerID" in processed_df.columns:
                    cust_ids = processed_df["customerID"]
                else:
                    cust_ids = pd.Series(range(len(processed_df)))

                if "TotalCharges" in processed_df.columns:
                    processed_df["TotalCharges"] = pd.to_numeric(processed_df["TotalCharges"], errors="coerce").fillna(0)

                for col, le in label_encoders.items():
                    if col in processed_df.columns:
                        processed_df[col] = processed_df[col].astype(str).map(
                            lambda s: le.transform([s])[0] if s in le.classes_ else 0
                        )

                features_only = processed_df[[c for c in feature_names if c in processed_df.columns]]
                batch_probs = model.predict_proba(features_only)[:, 1]
                batch_preds = (batch_probs >= 0.5).astype(int)

                batch_df["Churn_Prediction"] = np.where(batch_preds == 1, "High Risk", "Loyal")
                batch_df["Churn_Probability_%"] = np.round(batch_probs * 100, 2)

                st.dataframe(batch_df[["Churn_Prediction", "Churn_Probability_%"] + [c for c in batch_df.columns if c not in ["Churn_Prediction", "Churn_Probability_%"]]].head(10))
                
                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="DOWNLOAD RETENTION REPORT (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="churn_predictions_matrix.csv",
                    mime="text/csv"
                )
        st.markdown('</div>', unsafe_allow_html=True)