import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the trained PyCaret model
model = joblib.load('setbm.pkl')

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
        prediction_variance = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
        return prediction, prediction_variance

    if st.sidebar.button('Predict'):
        prediction_result, prediction_variance = predict_specific_energy(user_inputs)
        st.write('Predicted Specific Energy:', prediction_result)
        if prediction_variance is not None:
            st.write('Prediction Variance:', prediction_variance)


elif page == 'Visualization':
    st.title('Visualization of Model Performance')

    # Check if 'user_inputs' and 'prediction_result' are available
    if 'user_inputs' in st.session_state and 'prediction_result' in st.session_state:
        user_inputs = st.session_state['user_inputs']
        predicted_energy = st.session_state['prediction_result']

        # Plotting the PyCaret model's visualizations
        st.subheader('Model Visualizations')

        # Plot feature importance
        st.write('Feature Importance Plot')
        plot_model(model, plot='feature', verbose=False, display_format='streamlit')

        # Plot residuals
        st.write('Residuals Plot')
        plot_model(model, plot='residuals', verbose=False, display_format='streamlit')

        # Plot learning curve (if the model supports it)
        st.write('Learning Curve')
        try:
            plot_model(model, plot='learning', verbose=False,display_format='streamlit')
        except Exception as e:
            st.write("Learning Curve is not available for this model.")

    else:
        st.write("Please input TBM parameters in the 'Online' section first.")
