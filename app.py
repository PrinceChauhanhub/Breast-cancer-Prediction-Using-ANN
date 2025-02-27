import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Breast Cancer Prediction')

# user_input 

import streamlit as st

radius_mean = st.number_input('Radius Mean', min_value=6.981, max_value=28.11)
texture_mean = st.number_input('Texture Mean', min_value=9.71, max_value=39.28)
smoothness_mean = st.number_input('Smoothness Mean', min_value=0.05263, max_value=0.1634)
compactness_mean = st.number_input('Compactness Mean', min_value=0.01938, max_value=0.3454)
concavity_mean = st.number_input('Concavity Mean', min_value=0.0, max_value=0.4268)
concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, max_value=0.2012)
symmetry_mean = st.number_input('Symmetry Mean', min_value=0.106, max_value=0.304)
fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.04996, max_value=0.09744)
radius_se = st.number_input('Radius SE', min_value=0.1115, max_value=2.873)
texture_se = st.number_input('Texture SE', min_value=0.3602, max_value=4.885)
smoothness_se = st.number_input('Smoothness SE', min_value=0.001713, max_value=0.03113)
compactness_se = st.number_input('Compactness SE', min_value=0.002252, max_value=0.1354)
concavity_se = st.number_input('Concavity SE', min_value=0.0, max_value=0.396)
concave_points_se = st.number_input('Concave Points SE', min_value=0.0, max_value=0.05279)
symmetry_se = st.number_input('Symmetry SE', min_value=0.007882, max_value=0.07895)
fractal_dimension_se = st.number_input('Fractal Dimension SE', min_value=0.0008948, max_value=0.02984)
radius_worst = st.number_input('Radius Worst', min_value=7.93, max_value=36.04)
texture_worst = st.number_input('Texture Worst', min_value=12.02, max_value=49.54)
smoothness_worst = st.number_input('Smoothness Worst', min_value=0.07117, max_value=0.2226)
compactness_worst = st.number_input('Compactness Worst', min_value=0.02729, max_value=1.058)
concavity_worst = st.number_input('Concavity Worst', min_value=0.0, max_value=1.252)
concave_points_worst = st.number_input('Concave Points Worst', min_value=0.0, max_value=0.291)
symmetry_worst = st.number_input('Symmetry Worst', min_value=0.1565, max_value=0.6638)
fractal_dimension_worst = st.number_input('Fractal Dimension Worst', min_value=0.05504, max_value=0.2075)



input_data = pd.DataFrame({
    'radius_mean': [radius_mean],
    'texture_mean': [texture_mean],
    'smoothness_mean': [smoothness_mean],
    'compactness_mean': [compactness_mean],
    'concavity_mean': [concavity_mean],
    'concave points_mean': [concave_points_mean],
    'symmetry_mean': [symmetry_mean],
    'fractal_dimension_mean': [fractal_dimension_mean],
    'radius_se': [radius_se],
    'texture_se': [texture_se],
    'smoothness_se': [smoothness_se],
    'compactness_se': [compactness_se],
    'concavity_se': [concavity_se],
    'concave points_se': [concave_points_se],
    'symmetry_se': [symmetry_se],
    'fractal_dimension_se': [fractal_dimension_se],
    'radius_worst': [radius_worst],
    'texture_worst': [texture_worst],
    'smoothness_worst': [smoothness_worst],
    'compactness_worst': [compactness_worst],
    'concavity_worst': [concavity_worst],
    'concave points_worst': [concave_points_worst],
    'symmetry_worst': [symmetry_worst],
    'fractal_dimension_worst': [fractal_dimension_worst]
})
if st.button("Predict"):
    ## Scale the input data
    input_data_scaled = scaler.transform(input_data)

    ## Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]

    st.write(f"Probability of having cancer: {prediction_prob:.2f}")

    if prediction_prob > .5:
        st.write('Patient Diagnosed with Breast Cancer') 
    else:
        st.write("Patient not Diagnosed with Breast Cancer")
