import streamlit as st
import pandas as pd
import joblib

# Load the trained PyCaret model
model = joblib.load('setbm')

# Main Page Navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Choose a page', ['Online', 'Batch'])

if page == 'Online':
    st.title('TBM Specific Energy Prediction (Online)')
    
    # Online Data Input with sliders
    st.sidebar.header('Play with TBM Parameters')
    pressure_gauge_1 = st.sidebar.slider('Pressure Gauge 1 (kPa)', min_value=0, max_value=1000, value=500)
    pressure_gauge_2 = st.sidebar.slider('Pressure Gauge 2 (kPa)', min_value=0, max_value=1000, value=500)
    pressure_gauge_3 = st.sidebar.slider('Pressure Gauge 3 (kPa)', min_value=0, max_value=1000, value=500)
    pressure_gauge_4 = st.sidebar.slider('Pressure Gauge 4 (kPa)', min_value=0, max_value=1000, value=500)
    digging_velocity_left = st.sidebar.slider('Digging Velocity Left (mm/min)', min_value=0, max_value=1000, value=500)
    digging_velocity_right = st.sidebar.slider('Digging Velocity Right (mm/min)', min_value=0, max_value=1000, value=500)
    advancement_speed = st.sidebar.slider('Advancement Speed', min_value=0, max_value=100, value=50)
    shield_jack_stroke_left = st.sidebar.slider('Shield Jack Stroke Left (mm)', min_value=0, max_value=1000, value=500)
    shield_jack_stroke_right = st.sidebar.slider('Shield Jack Stroke Right (mm)', min_value=0, max_value=1000, value=500)
    propulsion_pressure = st.sidebar.slider('Propulsion Pressure (MPa)', min_value=0, max_value=10, value=5)
    total_thrust = st.sidebar.slider('Total Thrust (kN)', min_value=0, max_value=1000, value=500)
    cutter_torque = st.sidebar.slider('Cutter Torque (kN.m)', min_value=0, max_value=1000, value=500)
    cutterhead_rotation_speed = st.sidebar.slider('Cutterhead Rotation Speed (rpm)', min_value=0, max_value=1000, value=500)
    screw_pressure = st.sidebar.slider('Screw Pressure (MPa)', min_value=0, max_value=10, value=5)
    screw_rotation_speed = st.sidebar.slider('Screw Rotation Speed (rpm)', min_value=0, max_value=100, value=50)
    gate_opening = st.sidebar.slider('Gate Opening (%)', min_value=0, max_value=100, value=50)
    mud_injection_pressure = st.sidebar.slider('Mud Injection Pressure (MPa)', min_value=0, max_value=10, value=5)
    add_mud_flow = st.sidebar.slider('Add Mud Flow (L/min)', min_value=0, max_value=100, value=50)
    back_injection_rate = st.sidebar.slider('Back Injection Rate (%)', min_value=0, max_value=100, value=50)
    
    user_inputs = {
        'Pressure Gauge 1 (kPa)': pressure_gauge_1,
        'Pressure Gauge 2 (kPa)': pressure_gauge_2,
        'Pressure Gauge 3 (kPa)': pressure_gauge_3,
        'Pressure Gauge 4 (kPa)': pressure_gauge_4,
        'Digging Velocity Left (mm/min)': digging_velocity_left,
        'Digging Velocity Right (mm/min)': digging_velocity_right,
        'Advancement Speed': advancement_speed,
        'Shield Jack Stroke Left (mm)': shield_jack_stroke_left,
        'Shield Jack Stroke Right (mm)': shield_jack_stroke_right,
        'Propulsion Pressure (MPa)': propulsion_pressure,
        'Total Thrust (kN)': total_thrust,
        'Cutter Torque (kN.m)': cutter_torque,
        'Cutterhead Rotation Speed (rpm)': cutterhead_rotation_speed,
        'Screw Pressure (MPa)': screw_pressure,
        'Screw Rotation Speed (rpm)': screw_rotation_speed,
        'Gate Opening (%)': gate_opening,
        'Mud Injection Pressure (MPa)': mud_injection_pressure,
        'Add Mud Flow (L/min)': add_mud_flow,
        'Back Injection Rate (%)': back_injection_rate,
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
