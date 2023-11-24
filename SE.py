import streamlit as st
import pandas as pd
import joblib

# Load the trained PyCaret model
model = joblib.load('setbm.pkl')

# Main Page Navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Choose a page', ['Online', 'Batch'])

if page == 'Online':
    st.title('TBM Specific Energy Prediction (Online)')
    
    # Online Data Input with sliders
    st.sidebar.header('Play with TBM Parameters')
    cutterhead_rpm = st.sidebar.slider('Cutterhead RPM', min_value=0, max_value=1000, value=500)
    thrust_force = st.sidebar.slider('Thrust Force (kN)', min_value=0, max_value=500, value=250)
    
    user_inputs = {
        'Cutterhead RPM': cutterhead_rpm,
        'Thrust Force (kN)': thrust_force,
        # Add more parameters here as needed
    }

    def predict_specific_energy(operational_params):
        input_data = pd.DataFrame(operational_params, index=[0])
        prediction = model.predict(input_data)
        return prediction

    if st.sidebar.button('Predict'):
        prediction_result = predict_specific_energy(user_inputs)
        st.write('Predicted Specific Energy:', prediction_result)

elif page == 'Batch':
    st.title('TBM Specific Energy Prediction (Batch Upload)')
    
    # Batch Data Option - Upload CSV or Excel
    uploaded_file = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
            predictions = model.predict(data)
            st.write('Predicted Specific Energy for Uploaded Data:')
            st.write(predictions)
        except Exception as e:
            st.write('Error:', e)
