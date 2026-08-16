"""
EXAMPLE: Enhanced app.py with Results Interpretation & Population Stats Comparison
This shows what the predictor page could look like with interpretation features.
"""

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Telomere Length Estimator", layout="wide")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "intro"

# Load model
model = joblib.load("telomere_model.pkl")
feature_cols = joblib.load("model_features.pkl")

# Population statistics from notebook analysis
POPULATION_STATS = {
    'age_mean': 42.5,
    'age_range': (3, 85),
    'telomere_mean': 0.998,
    'telomere_std': 0.296,
    'telomere_range': (0.389, 9.42),
    'bmi_mean': 28.4,
    'bmi_range': (15.2, 66.4),
    'sample_size': 3567
}

# ==================== INTRO PAGE ====================
if st.session_state.page == "intro":
    st.markdown(
        """
        <style>
        .intro-container {
            text-align: center;
            padding: 40px 20px;
        }
        .intro-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 20px;
        }
        .intro-subtitle {
            font-size: 1.3em;
            color: #555;
            margin-bottom: 30px;
        }
        .intro-description {
            font-size: 1.1em;
            color: #666;
            line-height: 1.6;
            max-width: 700px;
            margin: 0 auto 40px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="intro-container">', unsafe_allow_html=True)
    st.markdown('<div class="intro-title">🧬 Telomere Length Estimator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="intro-subtitle">Predict Your Cellular Aging Profile</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-description">
        <p><strong>Welcome to this personal project!</strong></p>
        
        <p>Telomeres are protective caps on our DNA that shorten with age and stress. 
        Research shows telomere length is associated with biological aging and health outcomes.</p>
        
        <p><strong>What This Tool Does:</strong></p>
        <ul style="text-align: left; display: inline-block;">
            <li>Uses NHANES population data (3,567 participants) to build a predictive model</li>
            <li>Estimates your mean telomere length (T/S ratio) based on demographic factors</li>
            <li>Leverages machine learning trained on real-world health data</li>
        </ul>
        
        <p><strong>How It Works:</strong></p>
        <p>Simply enter your age, physical measurements (height & weight), and demographic information. 
        The model will provide an estimated telomere length score that you can compare against 
        the general population.</p>
        
        <p style="color: #888; font-size: 0.95em; margin-top: 30px;">
        <em>Note: This is an educational tool. Results are statistical estimates based on population trends, 
        not diagnostic assessments. Should not replace professional medical advice.</em>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Let's Get Started", key="start_btn", use_container_width=True):
            st.session_state.page = "predictor"
            st.rerun()

# ==================== PREDICTOR PAGE ====================
elif st.session_state.page == "predictor":
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Intro"):
            st.session_state.page = "intro"
            st.rerun()
    with col2:
        st.title("Telomere Length Estimator")

    # TAB 1: Predictor | TAB 2: Model Info
    tab1, tab2 = st.tabs(["🔮 Make Prediction", "📊 Model & Population"])

    with tab1:
        st.subheader("Enter Your Details")
        
        col1, col2 = st.columns(2)
        with col1:
            age_text = st.text_input("Age (years)", value="40")
            height_text = st.text_input("Height (cm)", value="170.0")
        with col2:
            weight_text = st.text_input("Weight (kg)", value="70.0")
            sex = st.radio("Sex", options=["Female", "Male"], horizontal=True)

        race_map = {
            "Mexican American": 1,
            "Other Hispanic": 2,
            "Non-Hispanic White": 3,
            "Non-Hispanic Black": 4,
            "Other / Multiracial": 5,
        }
        race_selected = st.selectbox("Race / Ethnicity", options=list(race_map.keys()))
        race_code = race_map[race_selected]

        if st.button("Calculate Prediction", type="primary", use_container_width=True):
            if not age_text.strip() or not height_text.strip() or not weight_text.strip():
                st.warning("⚠️ Please fill out all fields.")
            else:
                try:
                    age = int(age_text)
                    height_cm = float(height_text)
                    weight_kg = float(weight_text)
                    bmi = weight_kg / ((height_cm / 100) ** 2)

                    row = {
                        "RIDAGEYR": age,
                        "sex_female": 1 if sex == "Female" else 0,
                        "BMXBMI": bmi,
                        "cycle_2001": 0,
                        "race_2.0": 1 if race_code == 2 else 0,
                        "race_3.0": 1 if race_code == 3 else 0,
                        "race_4.0": 1 if race_code == 4 else 0,
                        "race_5.0": 1 if race_code == 5 else 0,
                    }

                    input_df = pd.DataFrame([row])[feature_cols]
                    prediction = model.predict(input_df)[0]

                    st.success(f"✅ Your Estimated Telomere Length (T/S Ratio): **{prediction:.3f}**")

                    # ==================== RESULTS INTERPRETATION ====================
                    st.markdown("---")
                    st.subheader("📈 What Does This Mean?")

                    # Calculate percentiles based on normal distribution
                    z_score = (prediction - POPULATION_STATS['telomere_mean']) / POPULATION_STATS['telomere_std']
                    percentile = int((1 + np.tanh(z_score / np.sqrt(2))) * 50)  # Approximate percentile

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Your T/S Ratio", f"{prediction:.3f}")
                    with col2:
                        st.metric("Population Mean", f"{POPULATION_STATS['telomere_mean']:.3f}")
                    with col3:
                        st.metric("vs. Population", f"{prediction - POPULATION_STATS['telomere_mean']:+.3f}")

                    # Interpretation
                    st.info(f"""
                    **Your Result Explained:**
                    
                    - Your telomere score places you in approximately the **{percentile}th percentile** of the population
                    - Average population T/S ratio: {POPULATION_STATS['telomere_mean']:.3f}
                    - Your difference from average: {prediction - POPULATION_STATS['telomere_mean']:+.3f} ({"shorter" if prediction < POPULATION_STATS['telomere_mean'] else "longer"} than average)
                    - The model explains ~17.5% of telomere variation (age & BMI are key factors)
                    """)

                    # ==================== POPULATION STATS COMPARISON ====================
                    st.markdown("---")
                    st.subheader("🌍 How You Compare to Population")

                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    with comp_col1:
                        st.write("**Your Profile**")
                        st.write(f"Age: {age}")
                        st.write(f"BMI: {bmi:.1f} kg/m²")
                        st.write(f"Sex: {sex}")
                        st.write(f"Race/Ethnicity: {race_selected}")

                    with comp_col2:
                        st.write("**Population Averages**")
                        st.write(f"Mean Age: {POPULATION_STATS['age_mean']:.1f}")
                        st.write(f"Mean BMI: {POPULATION_STATS['bmi_mean']:.1f} kg/m²")
                        st.write("Mixed (sample n=3,567)")
                        st.write("Mixed (NHANES data)")

                    with comp_col3:
                        st.write("**Population Ranges**")
                        st.write(f"Age: {POPULATION_STATS['age_range'][0]}-{POPULATION_STATS['age_range'][1]} yrs")
                        st.write(f"BMI: {POPULATION_STATS['bmi_range'][0]}-{POPULATION_STATS['bmi_range'][1]} kg/m²")
                        st.write(f"T/S: {POPULATION_STATS['telomere_range'][0]:.2f}-{POPULATION_STATS['telomere_range'][1]:.2f}")
                        st.write(f"Std Dev: ±{POPULATION_STATS['telomere_std']:.3f}")

                    # Visualization: Where you fall in distribution
                    st.markdown("---")
                    st.subheader("📊 Distribution Comparison")

                    # Create histogram-like comparison
                    x_range = np.linspace(POPULATION_STATS['telomere_range'][0], POPULATION_STATS['telomere_range'][1], 100)
                    from scipy.stats import norm
                    y_range = norm.pdf(x_range, POPULATION_STATS['telomere_mean'], POPULATION_STATS['telomere_std'])

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_range,
                        mode='lines',
                        name='Population Distribution',
                        line=dict(color='#1f77b4', width=2),
                        fill='tozeroy'
                    ))
                    fig.add_vline(
                        x=POPULATION_STATS['telomere_mean'],
                        line_dash="dash",
                        line_color="#ff7f0e",
                        annotation_text="Population Mean",
                        annotation_position="top"
                    )
                    fig.add_vline(
                        x=prediction,
                        line_dash="solid",
                        line_color="#2ca02c",
                        annotation_text=f"Your Score: {prediction:.3f}",
                        annotation_position="top"
                    )
                    fig.update_layout(
                        title="Your Telomere Length in Population Context",
                        xaxis_title="T/S Ratio",
                        yaxis_title="Density",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except ValueError:
                    st.error("❌ Please enter valid numbers for Age, Height, and Weight.")

    # ==================== TAB 2: Model Info ====================
    with tab2:
        st.subheader("🔬 Model & Data Information")

        st.markdown("**Model Performance**")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("Model R²", "0.175", "17.5% variance explained")
        with perf_col2:
            st.metric("Training Data", "3,567", "NHANES participants")
        with perf_col3:
            st.metric("Cross-Validation", "5-fold", "Robust testing")

        # Feature Importance
        st.markdown("**Feature Importance**")
        features_data = {
            'Age': 72.95,
            'BMI': 12.21,
            'Race': 8.33,
            'Sex': 1.32,
            'Other': 5.19
        }
        fig_feat = go.Figure(data=[
            go.Bar(x=list(features_data.values()), 
                   y=list(features_data.keys()),
                   orientation='h',
                   marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']))
        ])
        fig_feat.update_layout(
            xaxis_title="Importance (%)",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        st.markdown("""
        **Key Findings:**
        - Age is the dominant predictor (~73% of model importance)
        - BMI provides modest additional predictive value (~12%)
        - Behavioral factors (smoking, alcohol, activity) don't improve predictions
        - Individual telomere length varies widely (~±0.296 around mean)
        - Model limitations: ~83% of variation unexplained by measured factors
        
        **Biological Interpretation:**
        Telomere length is primarily determined by age and partially by metabolic factors (BMI). 
        Other unmeasured factors like genetic predisposition, chronic stress, inflammation, 
        and cellular senescence likely account for the remaining variation.
        """)

        st.warning("""
        **⚠️ Important Limitations:**
        - This is a statistical model, not a diagnostic tool
        - Results represent population-level trends, not individual biology
        - Cross-sectional data (NHANES) doesn't establish causation
        - Real telomere length requires laboratory measurement (qPCR)
        """)
