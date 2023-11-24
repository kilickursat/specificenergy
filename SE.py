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
    
    # Function to predict specific energy based on user inputs
    def predict_specific_energy(operational_params):
        input_data = pd.DataFrame(operational_params, index=[0])
        prediction = model.predict(input_data)
        return prediction

    # Make predictions using user inputs
    predicted_energy = predict_specific_energy(user_inputs)

    # Prediction Score Plot
    fig, ax = plt.subplots()
    sns.histplot(predicted_energy, kde=True, ax=ax)
    st.pyplot(fig)

    # Prediction Error Plot (Residuals)
    fig, ax = plt.subplots()
    sns.residplot(x=np.linspace(0, 1, len(predicted_energy)), y=predicted_energy, lowess=True, ax=ax)
    st.pyplot(fig)

    # Feature Importance (if available from your model)
    # Replace this with your actual feature importance calculation based on your_data
    # Replace 'feature_importance' with the actual feature importance data from your model
    # Assuming you have a model attribute 'feature_importances_' containing feature importance values
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=user_inputs.keys(), y=feature_importance, ax=ax)  # Plot feature importance
        ax.set_title('Feature Importance')
        ax.set_ylabel('Importance')
        ax.set_xlabel('Features')
        st.pyplot(fig)
    else:
        st.write("Feature importance is not available for this model.")

