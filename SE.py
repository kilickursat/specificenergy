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
    
    # Online Data Input with sliders
    st.sidebar.header('Play with TBM Parameters')
    params = {
        'Pressure Gauge 1 (kPa)': (0, 1000, 500),
        'Pressure Gauge 2 (kPa)': (0, 1000, 500),
        'Pressure Gauge 3 (kPa)': (0, 1000, 500),
        'Pressure Gauge 4 (kPa)': (0, 1000, 500),
        'Digging Velocity Left (mm/min)': (0, 1000, 500),
        'Digging Velocity Right (mm/min)': (0, 1000, 500),
        'Advancement Speed': (0, 100, 50),
        'Shield Jack Stroke Left (mm)': (0, 1000, 500),
        'Shield Jack Stroke Right (mm)': (0, 1000, 500),
        'Propulsion Pressure (MPa)': (0, 10, 5),
        'Total Thrust (kN)': (0, 1000, 500),
        'Cutter Torque (kN.m)': (0, 1000, 500),
        'Cutterhead Rotation Speed (rpm)': (0, 1000, 500),
        'Screw Pressure (MPa)': (0, 10, 5),
        'Screw Rotation Speed (rpm)': (0, 100, 50),
        'Gate Opening (%)': (0, 100, 50),
        'Mud Injection Pressure (MPa)': (0, 10, 5),
        'Add Mud Flow (L/min)': (0, 100, 50),
        'Back Injection Rate (%)': (0, 100, 50),
        # Add more parameters here as needed
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
    
    # Generate random data for visualization
    np.random.seed(42)
    sample_data = pd.DataFrame(np.random.rand(100, 5), columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
    sample_target = np.random.randint(0, 2, size=100)

    # Prediction Score Plot
    fig, ax = plt.subplots()
    sns.histplot(sample_target, kde=True, ax=ax)
    st.pyplot(fig)

    # Prediction Error Plot (Residuals)
    fig, ax = plt.subplots()
    sns.residplot(x=np.random.rand(100), y=sample_target, lowess=True, ax=ax)
    st.pyplot(fig)

    # Feature Importance
    fig, ax = plt.subplots()
    sns.barplot(x=sample_data.columns, y=np.abs(np.random.rand(5)), ax=ax)
    ax.set_title('Feature Importance')
    ax.set_ylabel('Importance')
    ax.set_xlabel('Features')
    st.pyplot(fig)
