import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# --- Load your trained model ---
model = pickle.load(open('Cancer_prediction_model_svm.pkl', 'rb'))

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

# --- Sidebar: Inputs ---
st.sidebar.header("🩺 Enter Patient Data")

def user_input_features():
    data = {
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
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Display input data ---
st.subheader("🧾 Patient Input Data")
st.dataframe(input_df)

# --- Define realistic ranges for out-of-range check ---
feature_ranges = {
    'radius_mean': (6.0, 30.0),
    'texture_mean': (9.0, 40.0),
    'perimeter_mean': (40.0, 190.0),
    'area_mean': (150.0, 2500.0),
    'smoothness_mean': (0.05, 0.2),
    'compactness_mean': (0.0, 0.35),
    'concavity_mean': (0.0, 0.4),
    'concave points_mean': (0.0, 0.2),
    'symmetry_mean': (0.1, 0.3),
    'fractal_dimension_mean': (0.0, 0.1),
    'radius_se': (0.1, 4.0),
    'texture_se': (0.3, 5.0),
    'perimeter_se': (0.5, 20.0),
    'area_se': (6.0, 500.0),
    'smoothness_se': (0.001, 0.05),
    'compactness_se': (0.0, 0.1),
    'concavity_se': (0.0, 0.2),
    'concave points_se': (0.0, 0.06),
    'symmetry_se': (0.0, 0.1),
    'fractal_dimension_se': (0.0, 0.03),
    'radius_worst': (7.0, 40.0),
    'texture_worst': (12.0, 50.0),
    'perimeter_worst': (50.0, 250.0),
    'area_worst': (200.0, 3000.0),
    'smoothness_worst': (0.05, 0.3),
    'compactness_worst': (0.0, 1.0),
    'concavity_worst': (0.0, 1.0),
    'concave points_worst': (0.0, 0.3),
    'symmetry_worst': (0.1, 0.5),
    'fractal_dimension_worst': (0.02, 0.2)
}

# --- Prediction ---
if st.button('🔍 Predict'):
    out_of_range = False
    for col in input_df.columns:
        min_val, max_val = feature_ranges[col]
        if input_df[col][0] < min_val or input_df[col][0] > max_val:
            out_of_range = True
            break

    
    elif:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.error("⚠️ The model predicts: **Cancer Detected (Malignant)**")
        else:
            st.success("✅ The model predicts: **No Cancer (Benign)**")

        # Probability chart
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
