import streamlit as st
import pickle
import numpy as np
import pandas as pd

def preprocess_features(df):
    """
    Preprocess the input data to match the features used during model training.
    """
    # One-hot encoding for Gender
    gender_encoded = pd.get_dummies(df['Gender'], prefix='Gender')
    df = pd.concat([df, gender_encoded], axis=1).drop(columns=['Gender'])

    # Rename Total_Proteins to match training data's column name
    df.rename(columns={'Total_Proteins': 'Total_Protiens'}, inplace=True)

    # Ensure column order matches the model's expectations
    feature_order = [
        'Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
        'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio',
        'Gender_Female', 'Gender_Male'
    ]

    # Fill missing columns with 0 if they don't exist in the input
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    return df[feature_order]

def load_model(file_path):
    """
    Load a trained model from a pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load Random Forest model
rf_model = load_model('E:\\streamlitfiles\\xgb_liver_model.pkl')

# Streamlit app
def main():
    st.title("Liver Disease Prediction")
    st.markdown("This app predicts the likelihood of liver disease based on user inputs.")

    # Collect user input
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=1.0, step=0.1)
    direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.3, step=0.1)
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=200, step=1)
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0, value=20, step=1)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0, value=30, step=1)
    total_proteins = st.number_input("Total Proteins", min_value=0.0, value=6.0, step=0.1)
    albumin = st.number_input("Albumin", min_value=0.0, value=3.0, step=0.1)
    albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.0, step=0.1)

    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Total_Bilirubin': [total_bilirubin],
        'Direct_Bilirubin': [direct_bilirubin],
        'Alkaline_Phosphotase': [alkaline_phosphotase],
        'Alamine_Aminotransferase': [alamine_aminotransferase],
        'Aspartate_Aminotransferase': [aspartate_aminotransferase],
        'Total_Proteins': [total_proteins],
        'Albumin': [albumin],
        'Albumin_and_Globulin_Ratio': [albumin_and_globulin_ratio]
    })

    processed_data = preprocess_features(input_data)

    # Predict
    if st.button("Predict"):
        if rf_model:
            prediction = rf_model.predict(processed_data)
            # Display result
            result = "Liver Disease" if prediction[0] == 1 else "No Liver Disease"
            st.success(f"Prediction: {result}")
        else:
            st.error("Model not available. Please check the model file.")

if __name__ == '__main__':
    main()
