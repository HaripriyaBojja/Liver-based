import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Load the pickle files
scaler = pickle.load(open('E:\\streamlitfiles\\scaler (2).pkl', 'rb'))
model_random_forest = pickle.load(open('E:\\streamlitfiles\\Random Forest.pkl', 'rb'))
# Load the default dataset
df = pd.read_csv('E:\\streamlitfiles\\liver.csv')
# App Title
st.title("Liver Disease Prediction🩺")
# Tabs for navigation
tabs = st.tabs(["🏠Home", "🔍Data Exploratory", "📊Model Performance", "🧑‍🔬Prediction"])

# Home Tab
with tabs[0]:
    st.header("Welcome to the Liver Disease Prediction App")
    st.write(df.head())
    st.markdown("## Features in the Liver Disease Prediction Dataset")
    st.markdown(
        """
        - **Age**: The age of the individual in years.
        - **Gender**: The gender of the individual (Male/Female).
        - **Total Bilirubin**: The total level of bilirubin in the blood (mg/dL). High levels indicate liver dysfunction.
        - **Direct Bilirubin**: The level of direct (conjugated) bilirubin in the blood (mg/dL). Elevated levels suggest bile excretion issues.
        - **Alkaline Phosphatase (ALP)**: An enzyme level in the blood (IU/L). High ALP levels indicate liver or bile duct problems.
        - **Alanine Aminotransferase (ALT)**: An enzyme that helps break down proteins in the liver (IU/L). Elevated ALT levels are indicative of liver damage.
        - **Aspartate Aminotransferase (AST)**: An enzyme found in the liver and other tissues (IU/L). High levels signal liver injury.
        - **Total Proteins**: The total amount of proteins in the blood (g/dL). Low levels indicate poor liver function.
        - **Albumin**: The level of albumin, a protein made by the liver (g/dL). Decreased levels indicate chronic liver disease.
        - **Albumin and Globulin Ratio (A/G Ratio)**: The ratio of albumin to globulin in the blood. Abnormal ratios reflect liver dysfunction.
        - **Dataset**: A binary indicator of liver disease (1 for presence, 0 for absence).
        """
    )
    st.markdown("## Count of Liver Disease and No Liver Disease Cases")
    liver_disease_counts = df['Dataset'].value_counts()
    st.write(liver_disease_counts)

    image_path = "E:\\streamlitfiles\\download.png"
    col1, col2, col3 = st.columns([1, 5, 4])
    with col2:
        st.markdown("Liver Disease vs No Liver Disease")
        if os.path.exists(image_path):
            st.image(image_path, caption="Liver Disease vs No Liver Disease", width=400)
        else:
            st.error("Error: The image file does not exist.")

# Data Exploratory Tab
with tabs[1]:
    st.header("Data Exploratory")
    st.write("Exploratory Data Analysis results can be displayed here.")
    st.write("Data Summary")
    st.write(df.describe())

    image_path = "E:\\streamlitfiles\\download2.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("Feature Distributions")
        if os.path.exists(image_path):
            st.image(image_path, caption="Feature Distributions", width=400)
        else:
            st.error("Error: The image file does not exist.")

    image_path = "E:\\streamlitfiles\\download4.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Correlation Heatmap")
        if os.path.exists(image_path):
            st.image(image_path, caption="Correlation Heatmap", width=400)
        else:
            st.error("Error: The image file does not exist.")

# Model Performance Tab
with tabs[2]:
    st.header("Model Performance")
    # Preprocess the data
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    X = df.drop('Dataset', axis=1)
    y = df['Dataset']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Generate predictions and classification report
    y_pred = model_random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Random Forest Classification Report")
    report_df = pd.DataFrame(classification_rep).transpose()
    st.dataframe(report_df)
    st.subheader("Accuracy")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    image_path = "E:\streamlitfiles\download5.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### confusion Matrix")
        if os.path.exists(image_path):
            st.image(image_path, caption="confusion Matrix", width=400)
        else:
            st.error("Error: The image file does not exist.")
    image_path = "E:\streamlitfiles\download6.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("Feature Importance for Liver Disease Prediction")
        if os.path.exists(image_path):
            st.image(image_path, caption="Feature Importance for Liver Disease Prediction", width=500)
        else:
            st.error("Error: The image file does not exist.") 
# Prediction Tab
with tabs[3]:
    st.header("Prediction")
    age = st.number_input("👶 Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("🚻 Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("🧪 Total Bilirubin", min_value=0.0, value=1.0, step=0.1)
    direct_bilirubin = st.number_input("🧪 Direct Bilirubin", min_value=0.0, value=0.3, step=0.1)
    alkaline_phosphotase = st.number_input("💉 Alkaline Phosphotase", min_value=0, value=200, step=1)
    alamine_aminotransferase = st.number_input("💉 Alamine Aminotransferase", min_value=0, value=20, step=1)
    aspartate_aminotransferase = st.number_input("💉 Aspartate Aminotransferase", min_value=0, value=30, step=1)
    total_proteins = st.number_input("🍽 Total Proteins", min_value=0.0, value=6.0, step=0.1)
    albumin = st.number_input("🧬 Albumin", min_value=0.0, value=3.0, step=0.1)
    albumin_and_globulin_ratio = st.number_input("🧬 Albumin and Globulin Ratio", min_value=0.0, value=1.0, step=0.1)
    input_features = np.array([
        [
            age,
            1 if gender == "Male" else 0,
            total_bilirubin,
            direct_bilirubin,
            alkaline_phosphotase,
            alamine_aminotransferase,
            aspartate_aminotransferase,
            total_proteins,
            albumin,
            albumin_and_globulin_ratio
        ]
    ])
    scaled_features = scaler.transform(input_features)
    if st.button("Predict"):
        prediction = model_random_forest.predict(scaled_features)[0]
        prediction_prob = model_random_forest.predict_proba(scaled_features)[0][1]
        result = "Liver Disease" if prediction == 1 else "No Liver Disease"
        st.write(f"### Prediction: {result}")
        st.write(f"### Prediction Probability: {prediction_prob:.2f}")
