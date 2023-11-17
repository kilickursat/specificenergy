import streamlit as st
import pandas as pd
from pycaret.regression import load_model

# Load the trained PyCaret model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    return load_model('path_to_your_model_file.pkl')

model = load_trained_model()

# SessionState to manage multiple pages
class SessionState:
    def __init__(self):
        self.page = None

# Initialize SessionState
session_state = SessionState()

# Set page layout to 'wide'
st.set_page_config(layout='wide')

# Define function to render the home page
def render_homepage():
    st.title('TBM Regression Model Prediction')
    st.write('Please select an option from the sidebar.')

# Function to get user input from sliders
def get_online_input():
    st.sidebar.header('Online Input Parameters')
    # Define sliders for online input
    parameter1 = st.sidebar.slider('Parameter 1', min_value=0.0, max_value=100.0, value=50.0)
    parameter2 = st.sidebar.slider('Parameter 2', min_value=0.0, max_value=100.0, value=50.0)
    
    online_input = {
        'Parameter 1': parameter1,
        'Parameter 2': parameter2
    }
    return pd.DataFrame(online_input, index=[0])

# Function to get user input from uploaded file
def get_batch_input():
    st.sidebar.header('Batch Input Data')
    uploaded_file = st.sidebar.file_uploader('Upload file', type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            batch_input = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            batch_input = pd.read_excel(uploaded_file)
        else:
            st.sidebar.warning('Uploaded file format not supported. Please upload a CSV or Excel file.')
            return None
        return batch_input
    return None

# Function to make predictions
def predict(input_data):
    if input_data is not None:
        prediction = model.predict(input_data)
        return prediction
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
    elif session_state.page == "Batch Input":
        batch_input_df = get_batch_input()
        if batch_input_df is not None:
            st.subheader('Batch Input Data:')
            st.write(batch_input_df)
            batch_prediction = predict(batch_input_df)
            if batch_prediction is not None:
                st.subheader('Batch Predictions:')
                st.write(batch_prediction)

# Create the Streamlit web app
def main():
    st.sidebar.title('Navigation')
    pages = ["Home", "Online Input", "Batch Input"]
    session_state.page = st.sidebar.radio("Go to", pages, index=0)

    render_pages()

if __name__ == '__main__':
    main()
