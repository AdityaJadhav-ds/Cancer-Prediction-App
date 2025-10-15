import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# --- Load trained model ---
try:
    with open('Cancer_prediction_model_svm.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found! Ensure 'Cancer_prediction_model_svm.pkl' is in the app directory.")
    st.stop()

# --- Page configuration ---
st.set_page_config(
    page_title="🧬 Cancer Prediction App",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Title & description ---
st.markdown("<h1 style='text-align:center; color:darkred;'>🧬 Cancer Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict whether a patient has <b>Cancer</b> based on medical input features.</p>", unsafe_allow_html=True)

# --- Sidebar input ---
st.sidebar.header("🩺 Enter Patient Data")

def user_input_features():
    data = {
        'radius_mean': st.sidebar.number_input('Radius Mean', 0.0, 50.0, 25.0),
        'texture_mean': st.sidebar.number_input('Texture Mean', 0.0, 50.0, 35.0),
        'perimeter_mean': st.sidebar.number_input('Perimeter Mean', 0.0, 200.0, 180.0),
        'area_mean': st.sidebar.number_input('Area Mean', 0.0, 3000.0, 2200.0),
        'smoothness_mean': st.sidebar.number_input('Smoothness Mean', 0.0, 1.0, 0.18),
        'compactness_mean': st.sidebar.number_input('Compactness Mean', 0.0, 1.0, 0.45),
        'concavity_mean': st.sidebar.number_input('Concavity Mean', 0.0, 1.0, 0.40),
        'concave points_mean': st.sidebar.number_input('Concave Points Mean', 0.0, 1.0, 0.18),
        'symmetry_mean': st.sidebar.number_input('Symmetry Mean', 0.0, 1.0, 0.30),
        'fractal_dimension_mean': st.sidebar.number_input('Fractal Dimension Mean', 0.0, 1.0, 0.09),

        'radius_se': st.sidebar.number_input('Radius SE', 0.0, 5.0, 2.0),
        'texture_se': st.sidebar.number_input('Texture SE', 0.0, 5.0, 3.5),
        'perimeter_se': st.sidebar.number_input('Perimeter SE', 0.0, 20.0, 15.0),
        'area_se': st.sidebar.number_input('Area SE', 0.0, 500.0, 350.0),
        'smoothness_se': st.sidebar.number_input('Smoothness SE', 0.0, 0.1, 0.04),
        'compactness_se': st.sidebar.number_input('Compactness SE', 0.0, 0.5, 0.09),
        'concavity_se': st.sidebar.number_input('Concavity SE', 0.0, 0.5, 0.18),
        'concave points_se': st.sidebar.number_input('Concave Points SE', 0.0, 0.2, 0.05),
        'symmetry_se': st.sidebar.number_input('Symmetry SE', 0.0, 0.5, 0.08),
        'fractal_dimension_se': st.sidebar.number_input('Fractal Dimension SE', 0.0, 0.1, 0.03),

        'radius_worst': st.sidebar.number_input('Radius Worst', 0.0, 50.0, 35.0),
        'texture_worst': st.sidebar.number_input('Texture Worst', 0.0, 50.0, 40.0),
        'perimeter_worst': st.sidebar.number_input('Perimeter Worst', 0.0, 200.0, 190.0),
        'area_worst': st.sidebar.number_input('Area Worst', 0.0, 3000.0, 2500.0),
        'smoothness_worst': st.sidebar.number_input('Smoothness Worst', 0.0, 1.0, 0.25),
        'compactness_worst': st.sidebar.number_input('Compactness Worst', 0.0, 1.0, 0.70),
        'concavity_worst': st.sidebar.number_input('Concavity Worst', 0.0, 1.0, 0.75),
        'concave points_worst': st.sidebar.number_input('Concave Points Worst', 0.0, 1.0, 0.25),
        'symmetry_worst': st.sidebar.number_input('Symmetry Worst', 0.0, 1.0, 0.40),
        'fractal_dimension_worst': st.sidebar.number_input('Fractal Dimension Worst', 0.0, 1.0, 0.12)
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Display input ---
st.subheader("🧾 Patient Input Data")
st.dataframe(input_df)

# --- Prediction ---
if st.button('🔍 Predict'):
    prediction = model.predict(input_df)[0]

    # --- Fix flipped labels: 0 = Benign, 1 = Malignant ---
    if hasattr(model, "predict_proba"):
        prediction_proba = model.predict_proba(input_df)[0]
        # Swap probability for correct label display
        proba_df = pd.DataFrame({
            'Condition': ['Benign', 'Malignant'],
            'Probability': [prediction_proba[0], prediction_proba[1]]
        })
        fig = px.bar(
            proba_df,
            x='Condition',
            y='Probability',
            color='Condition',
            color_discrete_sequence=['green', 'red'],
            text='Probability',
            title="Prediction Probabilities"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Prediction Result")
    # Correct label mapping
    if prediction == 'M':
        st.error("⚠️ The model predicts: **Cancer Detected (Malignant)**")
    else:
        st.success("✅ The model predicts: **No Cancer (Benign)**")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center;'>Created with ❤️ using <b>Python & Streamlit</b></p>", unsafe_allow_html=True)
