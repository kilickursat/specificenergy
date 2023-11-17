import streamlit as st
import pandas as pd
from pycaret.regression import load_model

# Set page layout to 'wide'
st.set_page_config(layout='wide')

# Load the trained PyCaret model
@st.cache_data
def load_trained_model():
    return load_model('specific-energy (1)')

model = load_trained_model()

# SessionState to manage multiple pages
class SessionState:
    def __init__(self):
        self.page = None

# Initialize SessionState
session_state = SessionState()

# Define function to render the home page
def render_homepage():
    st.title('TBM Regression Model Prediction')
    st.write('Please select an option from the sidebar.')

# Function to get user input from sliders based on provided parameters and descriptive statistics
def get_online_input():
    st.sidebar.header('Online Input Parameters')

    parameter_ranges = {
        'Pressure gauge 1 (kPa)': (5.0, 210.3),
        'Pressure gauge 2 (kPa)': (0.0, 259.0),
        'Pressure gauge 3 (kPa)': (0.0, 175.4),
        'Pressure gauge 4 (kPa)': (0.7, 426.9),
        'Digging velocity left (mm/min)': (0.0, 336.0),
        'Digging velocity right (mm/min)': (0.0, 239.0),
        'advancement speed': (0.0, 287.5),
        'Shield jack stroke left (mm)': (71.0, 3504.2),
        'Shield jack stroke right (mm)': (98.0, 5946.1),
        'Propulsion pressure (MPa)': (0.0, 31.5),
        'Total thrust (kN)': (0.0, 7160.7),
        'Cutter torque (kN.m)': (0.0, 694.9),
        'Cutterhead rotation speed (rpm)': (0.0, 7.0),
        'Screw pressure (MPa)': (-114.6, 7.7),
        'Screw rotation speed (rpm)': (-1.2, 78.3),
        'gate opening (%)': (0.0, 67.3),
        'Mud injection pressure (MPa)': (0.01, 1.6),
        'Add mud flow (L/min)': (0.0, 58.3),
        'Back in injection rate (%)': (0.0, 560.6)
    }

    

    online_input = {}
    for parameter, (min_val, max_val) in parameter_ranges.items():
        online_input[parameter] = st.sidebar.slider(parameter, min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)

    return pd.DataFrame(online_input, index=[0])

def predict(input_data):
    if input_data is not None:
        try:
            prediction = model.predict(input_data)
            return prediction
        except AttributeError as e:
            # Handle model access error
            if "get_feature_names" in str(e):
                error_msg = "Error accessing model features. Please ensure the model is loaded correctly."
                st.error(error_msg)
            else:
                raise e  # Re-raise the original error
    return None

# Function to render the different pages
def render_pages():
    if session_state.page == "Home":
        render_homepage()
    elif session_state.page == "Online Input":
        online_input_df = get_online_input()
        st.subheader('Online User Input:')
        st.write(online_input_df)
        online_prediction = predict(online_input_df)
        if online_prediction is not None:
            st.subheader('Online Prediction:')
            st.write(online_prediction)

# Create the Streamlit web app
def main():
    st.sidebar.title('Navigation')
    pages = ["Home", "Online Input"]
    session_state.page = st.sidebar.radio("Go to", pages, index=0)

    render_pages()

if __name__ == '__main__':
    main()
