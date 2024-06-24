import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Heart Disease Prediction App")
st.sidebar.title("User Input Features")

st.markdown("""
    This app uses a machine learning model to predict if a patient has heart disease based on their health parameters.
    Fill in the details in the sidebar to get the prediction.
""")

def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')
    sex = st.sidebar.selectbox('Sex', (0, 1))
    cp = st.sidebar.selectbox('Chest pain type', (0, 1, 2, 3))
    tres = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholesterol in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar', (0, 1))
    res = st.sidebar.number_input('Resting electrocardiographic results: ')
    tha = st.sidebar.number_input('Maximum heart rate achieved: ')
    exa = st.sidebar.selectbox('Exercise induced angina', (0, 1))
    old = st.sidebar.number_input('Oldpeak: ')
    slope = st.sidebar.number_input('The slope of the peak exercise ST segment: ')
    ca = st.sidebar.selectbox('Number of major vessels', (0, 1, 2, 3))
    thal = st.sidebar.selectbox('Thal', (0, 1, 2))

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': tres,
        'chol': chol,
        'fbs': fbs,
        'restecg': res,
        'thalach': tha,
        'exang': exa,
        'oldpeak': old,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Parameters')
st.write(input_df)

try:
    load_clf = joblib.load('Random_forest_model.joblib')

    prediction = load_clf.predict(input_df)
    prediction_proba = load_clf.predict_proba(input_df)

    st.subheader('Prediction')
    heart_disease = np.array(['No Disease', 'Disease'])
    st.write(heart_disease[prediction][0])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

except FileNotFoundError:
    st.error("Model file not found. Please ensure the model file 'Random_forest_model.joblib' is in the same directory as this script.")
except Exception as e:
    st.error(f"An error occurred: {e}")

st.sidebar.info("Fill in the parameters to get the prediction.")
st.sidebar.markdown("[Learn more about heart disease](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))")

st.markdown("""
    <style>
    .reportview-container .main footer {visibility: hidden;}
    .reportview-container .main:after {
        content: 'Developed by Your Name';
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
    }
    </style>
    """, unsafe_allow_html=True)