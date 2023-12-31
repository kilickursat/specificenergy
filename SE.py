import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pycaret.regression import *

model = joblib.load('TBM-energy.pkl')

# Main Page Navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Choose a page', ['Online', 'Visualization'])

if page == 'Online':
    st.title('TBM Specific Energy Prediction (Online)')

    # Create a dictionary mapping variable names to strings
    params = {
        'Pressure gauge 1 (kPa)': (0, 1000, 500),
        'Pressure gauge 2 (kPa)': (0, 1000, 500),
        'Pressure gauge 3 (kPa)': (0, 1000, 500),
        'Pressure gauge 4 (kPa)': (0, 1000, 500),
        'Digging velocity left (mm/min)': (0, 1000, 500),
        'Digging velocity right (mm/min)': (0, 1000, 500),
        'advancement speed ': (0, 100, 50),
        'Shield jack stroke left (mm)': (0, 1000, 500),
        'Shield jack stroke rigth (mm)': (0, 1000, 500),
        'Propulsion pressure (MPa)': (0, 10, 5),
        'Total thrust (kN)': (0, 1000, 500),
        'Cutter torque (kN.m)': (0, 1000, 500),
        'Cutterhead rotation speed (rpm)': (0, 1000, 500),
        'Screw pressure (MPa)': (0, 10, 5),
        'Screw rotation speed (rpm)': (0, 100, 50),
        'gate opening (%)': (0, 100, 50),
        'Mud injection pressure (MPa)': (0, 10, 5),
        'Add mud flow (L/min)': (0, 100, 50),
        'Back in injection rate (%)': (0, 100, 50),
    }

    user_inputs = {}
    for param, (min_val, max_val, default_val) in params.items():
        user_inputs[param] = st.sidebar.slider(param, min_value=min_val, max_value=max_val, value=default_val)

    def predict_specific_energy(operational_params):
        input_data = pd.DataFrame(operational_params, index=[0])
        prediction = model.predict(input_data)
        prediction_variance = (
            model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
        )
        return prediction, prediction_variance

    # Model visualizations moved outside the function
    st.subheader('Model Visualizations')
    
    try:
        st.write('Feature Importance Plot')
        fig = plot_model(prediction_result, plot='feature', verbose=False, display_format="streamlit")
        if fig:
            st.pyplot(fig)
        else:
            st.write("Feature Importance Plot is not available for this model.")
    except ValueError as e:
        st.write("Feature Importance Plot is not available for this model.")
    
    try:
        st.write('Residuals Plot')
        fig = plot_model(prediction_result, plot='residuals', verbose=False, display_format="streamlit")
        if fig:
            st.pyplot(fig)
        else:
            st.write("Residuals Plot is not available for this model.")
    except ValueError as e:
        st.write("Residuals Plot is not available for this model.")
    
    try:
        st.write('Learning Curve')
        fig = plot_model(prediction_result, plot='learning', verbose=False, display_format="streamlit")
        if fig:
            st.pyplot(fig)
        else:
            st.write("Learning Curve is not available for this model.")
    except ValueError as e:
        st.write("Learning Curve is not available for this model.")

    if st.sidebar.button('Predict'):
        prediction_result, prediction_variance = predict_specific_energy(user_inputs)
        st.write('Predicted Specific Energy:', prediction_result)
        if prediction_variance is not None:
            st.write('Prediction Variance:', prediction_variance)
