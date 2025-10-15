import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# --- Load trained model and scaler ---
model = pickle.load(open('Cancer_prediction_model_nb.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Make sure you save the scaler during training

# --- Page configuration ---
st.set_page_config(
    page_title="🧬 Cancer Prediction App",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- App Title ---
st.markdown(
    "<h1 style='text-align: center; color: darkblue;'>🧬 Cancer Prediction App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Predict whether a patient has <b>Cancer</b> based on medical input features.</p>",
    unsafe_allow_html=True
)

# --- Sidebar: User Inputs ---
st.sidebar.header("🩺 Enter Patient Data")

def user_input_features():
    feature_data = {
        'radius_mean': st.sidebar.number_input('Radius Mean', 0.0, 50.0, 14.0),
        'texture_mean': st.sidebar.number_input('Texture Mean', 0.0, 50.0, 19.0),
        'perimeter_mean': st.sidebar.number_input('Perimeter Mean', 0.0, 200.0, 90.0),
        'area_mean': st.sidebar.number_input('Area Mean', 0.0, 3000.0, 600.0),
        'smoothness_mean': st.sidebar.number_input('Smoothness Mean', 0.0, 1.0, 0.1),
        'compactness_mean': st.sidebar.number_input('Compactness Mean', 0.0, 1.0, 0.2),
        'concavity_mean': st.sidebar.number_input('Concavity Mean', 0.0, 1.0, 0.1),
        'concave points_mean': st.sidebar.number_input('Concave Points Mean', 0.0, 1.0, 0.05),
        'symmetry_mean': st.sidebar.number_input('Symmetry Mean', 0.0, 1.0, 0.2),
        'fractal_dimension_mean': st.sidebar.number_input('Fractal Dimension Mean', 0.0, 1.0, 0.06),
        'radius_se': st.sidebar.number_input('Radius SE', 0.0, 5.0, 0.3),
        'texture_se': st.sidebar.number_input('Texture SE', 0.0, 5.0, 1.0),
        'perimeter_se': st.sidebar.number_input('Perimeter SE', 0.0, 20.0, 2.0),
        'area_se': st.sidebar.number_input('Area SE', 0.0, 500.0, 20.0),
        'smoothness_se': st.sidebar.number_input('Smoothness SE', 0.0, 0.1, 0.01),
        'compactness_se': st.sidebar.number_input('Compactness SE', 0.0, 0.5, 0.02),
        'concavity_se': st.sidebar.number_input('Concavity SE', 0.0, 0.5, 0.02),
        'concave points_se': st.sidebar.number_input('Concave Points SE', 0.0, 0.2, 0.01),
        'symmetry_se': st.sidebar.number_input('Symmetry SE', 0.0, 0.5, 0.02),
        'fractal_dimension_se': st.sidebar.number_input('Fractal Dimension SE', 0.0, 0.1, 0.01),
        'radius_worst': st.sidebar.number_input('Radius Worst', 0.0, 50.0, 16.0),
        'texture_worst': st.sidebar.number_input('Texture Worst', 0.0, 50.0, 25.0),
        'perimeter_worst': st.sidebar.number_input('Perimeter Worst', 0.0, 200.0, 100.0),
        'area_worst': st.sidebar.number_input('Area Worst', 0.0, 3000.0, 800.0),
        'smoothness_worst': st.sidebar.number_input('Smoothness Worst', 0.0, 1.0, 0.15),
        'compactness_worst': st.sidebar.number_input('Compactness Worst', 0.0, 1.0, 0.3),
        'concavity_worst': st.sidebar.number_input('Concavity Worst', 0.0, 1.0, 0.3),
        'concave points_worst': st.sidebar.number_input('Concave Points Worst', 0.0, 1.0, 0.1),
        'symmetry_worst': st.sidebar.number_input('Symmetry Worst', 0.0, 1.0, 0.25),
        'fractal_dimension_worst': st.sidebar.number_input('Fractal Dimension Worst', 0.0, 1.0, 0.08)
    }
    features = pd.DataFrame(feature_data, index=[0])
    return features

input_df = user_input_features()

# --- Display Input Data ---
st.subheader("🧾 Patient Input Data")
st.dataframe(input_df)

# --- Prediction ---
if st.button('🔍 Predict'):
    # 1️⃣ Scale input
    input_scaled = scaler.transform(input_df)

    # 2️⃣ Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error("⚠️ The model predicts: **Cancer Detected (Malignant)**")
    else:
        st.success("✅ The model predicts: **No Cancer (Benign)**")

    # 3️⃣ Probability chart
    proba_df = pd.DataFrame({
        'Condition': ['Benign', 'Malignant'],
        'Probability': prediction_proba
    })
    fig = px.bar(
        proba_df,
        x='Condition',
        y='Probability',
        color='Probability',
        color_continuous_scale='RdBu',
        text='Probability',
        title="Prediction Probabilities"
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Created with ❤️ using <b>Python & Streamlit</b></p>",
    unsafe_allow_html=True
)
