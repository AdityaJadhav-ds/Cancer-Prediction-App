import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# --- Load the trained model ---
model = pickle.load(open('Cancer_prediction_model_nb.pkl', 'rb'))

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

# --- Sidebar for user input ---
st.sidebar.header("🩺 Enter Patient Data")

def user_input_features():
    # Collect user input
    mean_radius = st.sidebar.slider('Mean Radius', 0.0, 50.0, 14.0)
    mean_texture = st.sidebar.slider('Mean Texture', 0.0, 50.0, 19.0)
    mean_perimeter = st.sidebar.slider('Mean Perimeter', 0.0, 200.0, 90.0)
    mean_area = st.sidebar.slider('Mean Area', 0.0, 3000.0, 600.0)
    mean_smoothness = st.sidebar.slider('Mean Smoothness', 0.0, 1.0, 0.1)

    data = {
        'Mean Radius': mean_radius,
        'Mean Texture': mean_texture,
        'Mean Perimeter': mean_perimeter,
        'Mean Area': mean_area,
        'Mean Smoothness': mean_smoothness
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Display user input in a nice table ---
st.subheader("🧾 Patient Input Data")
st.dataframe(input_df.style.background_gradient(cmap='Blues'))

# --- Prediction ---
if st.button('🔍 Predict'):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Result")

    # Display result with a clear, colored alert
    if prediction == 1:
        st.error("⚠️ The model predicts: **Cancer Detected (Malignant)**")
    else:
        st.success("✅ The model predicts: **No Cancer (Benign)**")

    # Display probability as bar chart
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
