import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="Telomere Length Estimator", layout="wide")

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "intro"

# Load saved model and feature list
model = joblib.load("telomere_model.pkl")
feature_cols = joblib.load("model_features.pkl")

# Population statistics from NHANES analysis
POPULATION_STATS = {
    'age_mean': 42.5,
    'age_range': (3, 85),
    'telomere_mean': 0.998,
    'telomere_std': 0.296,
    'telomere_range': (0.389, 9.42),
    'bmi_mean': 28.4,
    'bmi_range': (15.2, 66.4),
    'sample_size': 3567,
    'model_r2': 0.175
}

# Function to classify BMI
def get_bmi_classification(bmi):
    """Classify BMI into categories"""
    if bmi < 18.5:
        return "Underweight", "🔵 Underweight"
    elif 18.5 <= bmi < 25.0:
        return "Normal Weight", "🟢 Normal Weight"
    elif 25.0 <= bmi < 30.0:
        return "Overweight", "🟡 Overweight"
    else:
        return "Obese", "🔴 Obese"

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
        <p><strong>Welcome to this personal project by Xuan!</strong></p>
        
        <p>Telomeres are protective caps on our DNA that shorten with age and stress. 
        Research shows telomere length is associated with biological aging and health outcomes.</p>
        
        <p><strong>What This Tool Does:</strong></p>
        <ul style="text-align: left; display: inline-block;">
            <li>Uses NHANES population data (3,567 participants) to build a predictive model</li>
            <li>Estimates your mean telomere length (T/S ratio) based on demographic factors</li>
            <li>Leverages machine learning trained on real-world health data</li>
            <li>Compares your telomere profile against population statistics</li>
        </ul>
        
        <p><strong>How It Works:</strong></p>
        <p>Simply enter your age, physical measurements (height & weight), and demographic information. 
        The model will provide an estimated telomere length score that you can compare against 
        the general population. Results show what your biological profile suggests about cellular aging.</p>
        
        <p><strong>Key Findings from the Model:</strong></p>
        <ul style="text-align: left; display: inline-block;">
            <li><strong>Age dominates:</strong> Account for ~73% of telomere variation</li>
            <li><strong>BMI matters:</strong> Body composition adds ~12% predictive value</li>
            <li><strong>Demographic factors:</strong> Race/ethnicity contribute modestly</li>
            <li><strong>Model explains 17.5%</strong> of telomere length variation</li>
        </ul>
        
        <p style="color: #888; font-size: 0.95em; margin-top: 30px;">
        <em>Note: This is an educational tool based on statistical models trained on NHANES data (1999-2000). 
        Results are population estimates, not diagnostic assessments. Should not replace professional medical advice.</em>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Create centered button
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

    # Create tabs for Prediction and Model Info
    tab1, tab2 = st.tabs(["Make Prediction", "Model & Population"])

    # ==================== TAB 1: PREDICTOR ====================
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

        if st.button("🧬 Calculate Prediction", type="primary", use_container_width=True):
            if not age_text.strip() or not height_text.strip() or not weight_text.strip():
                st.warning("Please fill out all fields before submitting.")
            else:
                try:
                    # Parse inputs
                    age = int(age_text)
                    height_cm = float(height_text)
                    weight_kg = float(weight_text)
                    bmi = weight_kg / ((height_cm / 100) ** 2)

                    # Build input for model
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

                    # Display main result with BMI classification
                    st.success(f"Your Estimated Telomere Length (T/S Ratio): **{prediction:.3f}**")
                    
                    bmi_category, bmi_emoji = get_bmi_classification(bmi)
                    st.caption(f"Calculated BMI: **{bmi:.1f} kg/m²** | {bmi_emoji}")

                    # ==================== RESULTS INTERPRETATION ====================
                    st.markdown("---")
                    st.subheader("📈 What Does This Mean?")

                    # Calculate percentile and comparison
                    z_score = (prediction - POPULATION_STATS['telomere_mean']) / POPULATION_STATS['telomere_std']
                    percentile = int((1 + np.tanh(z_score / np.sqrt(2))) * 50)  # Approximate percentile
                    diff_from_mean = prediction - POPULATION_STATS['telomere_mean']

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Your T/S Ratio", f"{prediction:.3f}")
                    with col2:
                        st.metric("Population Mean", f"{POPULATION_STATS['telomere_mean']:.3f}")
                    with col3:
                        direction = "Shorter ↓" if diff_from_mean < 0 else "Longer ↑"
                        st.metric("vs. Population", f"{diff_from_mean:+.3f}", direction)

                    # Interpretation box
                    if prediction < POPULATION_STATS['telomere_mean'] - POPULATION_STATS['telomere_std']:
                        interpretation = "**Shorter than average** (lower percentile) - May suggest more cellular aging relative to population"
                        color = "orange"
                    elif prediction > POPULATION_STATS['telomere_mean'] + POPULATION_STATS['telomere_std']:
                        interpretation = "**Longer than average** (higher percentile) - May suggest less cellular aging relative to population"
                        color = "green"
                    else:
                        interpretation = "**Near population average** - Typical telomere length for this demographic profile"
                        color = "blue"

                    st.info(f"""
                    **Your Result Explained:**
                    
                    - Your telomere score: **{interpretation}**
                    - Percentile rank: **~{percentile}th percentile**
                    - Distance from mean: **{diff_from_mean:+.3f}** standard deviations
                    - Population average: **{POPULATION_STATS['telomere_mean']:.3f}** (±{POPULATION_STATS['telomere_std']:.3f})
                    """)

                    # ==================== POPULATION STATS COMPARISON ====================
                    st.markdown("---")
                    st.subheader("🌍 How You Compare to Population")

                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    with comp_col1:
                        st.write("**Your Profile**")
                        st.write(f"• Age: **{age}** years")
                        st.write(f"• BMI: **{bmi:.1f}** kg/m² ({bmi_category})")
                        st.write(f"• Sex: **{sex}**")
                        st.write(f"• Race/Ethnicity: **{race_selected}**")

                    with comp_col2:
                        st.write("**Population Averages**")
                        st.write(f"• Mean Age: **{POPULATION_STATS['age_mean']:.1f}** years")
                        st.write(f"• Mean BMI: **{POPULATION_STATS['bmi_mean']:.1f}** kg/m²")
                        st.write(f"• T/S Ratio: **{POPULATION_STATS['telomere_mean']:.3f}**")
                        st.write(f"• Sample: **{POPULATION_STATS['sample_size']}** NHANES participants")

                    with comp_col3:
                        st.write("**Population Ranges**")
                        st.write(f"• Age: **{POPULATION_STATS['age_range'][0]}-{POPULATION_STATS['age_range'][1]}** years")
                        st.write(f"• BMI: **{POPULATION_STATS['bmi_range'][0]:.1f}-{POPULATION_STATS['bmi_range'][1]:.1f}** kg/m²")
                        st.write(f"• T/S Ratio: **{POPULATION_STATS['telomere_range'][0]:.2f}-{POPULATION_STATS['telomere_range'][1]:.2f}**")
                        st.write(f"• Std Deviation: **±{POPULATION_STATS['telomere_std']:.3f}**")

                    # ==================== DISTRIBUTION VISUALIZATION ====================
                    st.markdown("---")
                    st.subheader("📊 Your Result in Population Context")

                    # Create normal distribution with user's score
                    x_range = np.linspace(
                        POPULATION_STATS['telomere_range'][0],
                        POPULATION_STATS['telomere_range'][1],
                        200
                    )
                    y_range = norm.pdf(x_range, POPULATION_STATS['telomere_mean'], POPULATION_STATS['telomere_std'])

                    fig = go.Figure()
                    
                    # Population distribution
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_range,
                        mode='lines',
                        name='Population Distribution',
                        line=dict(color='#1f77b4', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(31, 119, 180, 0.2)'
                    ))
                    
                    # Population mean line
                    fig.add_vline(
                        x=POPULATION_STATS['telomere_mean'],
                        line_dash="dash",
                        line_color="#ff7f0e",
                        line_width=2,
                        annotation_text=f"Population Mean: {POPULATION_STATS['telomere_mean']:.3f}",
                        annotation_position="top left"
                    )
                    
                    # User's score line
                    fig.add_vline(
                        x=prediction,
                        line_dash="solid",
                        line_color="#2ca02c",
                        line_width=3,
                        annotation_text=f"Your Score: {prediction:.3f}",
                        annotation_position="top right"
                    )

                    fig.update_layout(
                        title="Your Telomere Length in Population Distribution",
                        xaxis_title="T/S Ratio (Telomere Length)",
                        yaxis_title="Density",
                        height=400,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except ValueError:
                    st.error("❌ Please enter valid numbers for Age, Height, and Weight.")

    # ==================== TAB 2: MODEL INFO ====================
    with tab2:
        st.subheader("Model Performance & Feature Importance")

        # Model performance metrics
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("Model R²", f"{POPULATION_STATS['model_r2']:.3f}", "% Variance Explained: 17.5%")
        with perf_col2:
            st.metric("Training Data", f"{POPULATION_STATS['sample_size']:,}", "NHANES Participants")
        with perf_col3:
            st.metric("Validation Method", "5-Fold CV", "Robust Testing")

        st.markdown("---")

        # Feature Importance Chart
        st.subheader("Feature Importance Ranking")
        features_data = {
            'Age': 72.95,
            'BMI': 12.21,
            'Race/Ethnicity': 8.33,
            'Sex': 1.32,
            'Other': 5.19
        }
        
        fig_feat = go.Figure(data=[
            go.Bar(
                x=list(features_data.values()), 
                y=list(features_data.keys()),
                orientation='h',
                marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),
                text=[f"{v:.1f}%" for v in features_data.values()],
                textposition='auto'
            )
        ])
        fig_feat.update_layout(
            xaxis_title="Importance (%)",
            height=300,
            showlegend=False,
            margin=dict(l=150)
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        st.markdown("---")
        st.subheader("Key Findings")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **What Works:**
            - Age is the dominant predictor (~73% importance)
            - BMI provides modest predictive value (~12%)
            - Demographic factors add ~8%
            - Model is stable across cross-validation
            """)
        
        with col2:
            st.markdown("""
            **Limitations:**
            - Explains only 17.5% of telomere variation
            - ~82.5% unexplained by measured factors
            - Behavioral factors (smoking, activity) don't improve predictions
            - Cross-sectional data (NHANES 1999-2000)
            """)

        st.markdown("---")
        st.subheader("🔍 Biological Interpretation")

        st.markdown("""
        **Why Age Dominates:**
        Telomeres naturally shorten over time as cells divide. This is the primary biological driver of variation 
        in the population. Age alone explains most observable differences in telomere length.

        **Why BMI Matters:**
        Body composition may reflect metabolic health and inflammatory status, which can affect cellular senescence. 
        Higher BMI is associated with shorter telomeres in this population.

        **Why Behavioral Factors Don't Help:**
        In this cross-sectional snapshot, lifestyle variables (smoking, activity, alcohol) don't improve predictions. 
        This suggests either: (1) weak effects in this population, (2) measurement error, or (3) other unmeasured factors dominate.

        **What's Missing:**
        The unexplained 82.5% likely includes genetic predisposition, chronic stress/cortisol, 
        inflammation markers, sleep quality, environmental exposures, and other biological factors 
        not captured in NHANES demographic data.
        """)

        st.markdown("---")
        st.subheader("Important Disclaimers")

        st.warning("""
        - **Not a diagnostic tool:** This model is for education and research purposes only
        - **Population estimates:** Results reflect statistical trends, not individual biology
        - **Cross-sectional data:** NHANES data from 1999-2000; association ≠ causation
        - **Real measurement:** Actual telomere length requires qPCR laboratory testing
        - **Individual variation:** High variance means predictions have wide confidence intervals
        - **Not medical advice:** Always consult healthcare professionals for health decisions
        """)
