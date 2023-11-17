import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

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
    if add_selectbox == "Online":
        Pressure gauge 1 (kPa) = st.sidebar.slider ('Pressure gauge 1 (kPa)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Pressure gauge 2 (kPa) = st.sidebar.slider ('Pressure gauge 2 (kPa)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Pressure gauge 3 (kPa) = st.sidebar.slider ('Pressure gauge 3 (kPa)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Pressure gauge 4 (kPa) = st.sidebar.slider ('Pressure gauge 4 (kPa)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Digging velocity left (mm/min) = st.sidebar.slider ('Digging velocity left (mm/min)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Digging velocity right (mm/min) = st.sidebar.slider ('Digging velocity right (mm/min)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        advancement speed = st.sidebar.slider ('advancement speed', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Shield jack stroke left (mm) = st.sidebar.slider ('Shield jack stroke left (mm)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Shield jack stroke right (mm) = st.sidebar.slider ('Shield jack stroke right (mm)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Total thrust (kN) = st.sidebar.slider ('Total thrust (kN)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Cutter torque (kN.m) = st.sidebar.slider ('Cutter torque (kN.m)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Cutterhead rotation speed (rpm) = st.sidebar.slider ('Cutterhead rotation speed (rpm)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Screw pressure (MPa) = st.sidebar.slider ('Pressure gauge 1 (kPa)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Screw rotation speed (rpm) = st.sidebar.slider ('Screw rotation speed (rpm)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        gate opening (%) = st.sidebar.slider ('gate opening (%)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Mud injection pressure (MPa)= st.sidebar.slider ('Mud injection pressure (MPa)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Add mud flow (L/min) = st.sidebar.slider ('Add mud flow (L/min)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
        Back in injection rate (%) = st.sidebar.slider ('Back in injection rate (%)', min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)

    input_dict = {'Pressure gauge 1 (kPa)': Pressure gauge 1 (kPa),
        'Pressure gauge 2 (kPa)':Pressure gauge 2 (kPa) ,
        'Pressure gauge 3 (kPa)': Pressure gauge 3 (kPa),
        'Pressure gauge 4 (kPa)':Pressure gauge 4 (kPa) ,
        'Digging velocity left (mm/min)': Digging velocity left (mm/min),
        'Digging velocity right (mm/min)':Digging velocity right (mm/min) ,
        'advancement speed':advancement speed ,
        'Shield jack stroke left (mm)':Shield jack stroke left (mm) ,
        'Shield jack stroke right (mm)':Shield jack stroke right (mm) ,
        'Propulsion pressure (MPa)':Propulsion pressure (MPa) ,
        'Total thrust (kN)': Total thrust (kN),
        'Cutter torque (kN.m)':Cutter torque (kN.m) ,
        'Cutterhead rotation speed (rpm)':Cutterhead rotation speed (rpm) ,
        'Screw pressure (MPa)':Screw pressure (MPa) ,
        'Screw rotation speed (rpm)':Screw rotation speed (rpm) ,
        'gate opening (%)': gate opening (%),
        'Mud injection pressure (MPa)':Mud injection pressure (MPa) ,
        'Add mud flow (L/min)': Add mud flow (L/min),
        'Back in injection rate (%)':Back in injection rate (%) }

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = '$' + str(output)

        st.success('The output is {}'.format(output))

# Function to make predictions
def predict(input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

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
