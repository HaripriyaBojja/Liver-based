import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# Load the pickle files
scaler = pickle.load(open('scaler (3).pkl', 'rb'))
with open('xgb_liver_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
# Load the default dataset
df = pd.read_csv('liver.csv')
# App Title
st.markdown("""
    <h1 style='text-align: center; color: white; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; font-size: 50px;'>
        🩺 Liver Disease Prediction
    </h1>
""", unsafe_allow_html=True)

# Tabs for navigation
tabs = st.tabs(["🏠Home", "🔍Data Exploratory", "📊Model Performance", "🧑‍🔬Prediction"])
st.markdown("""
    <style>
    /* Center the entire tab navigation */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        padding: 10px 0;
        gap: 40px;  /* spacing between tabs */
    }

    /* Style for each individual tab */
    .stTabs [data-baseweb="tab"] {
        font-size: 20px;
        font-weight: 600;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f0f0f0;
        padding: 10px 25px;
        border-radius: 12px;
        color: #000000;
        transition: background-color 0.3s ease;
    }

    /* Active (selected) tab styling */
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Home Tab
with tabs[0]:
    st.header("Welcome to the Liver Disease Prediction App")
    # Show styled and centered df.head()
    df_head = df.head()
# Custom CSS to style the DataFrame
    st.markdown("""
    <style>
    .centered-table {
        border-collapse: collapse;
        margin: auto;
        font-size: 18px;
        font-family: Arial, sans-serif;
        min-width: 600px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .centered-table thead tr {
        background-color: #2c3e50;
        color: #ffffff;
        text-align: center;
    }
    .centered-table th,
    .centered-table td {
    padding: 4px 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Render the styled table
    st.markdown(
    df_head.to_html(index=False, classes='centered-table'),
    unsafe_allow_html=True
)
    st.markdown("## Count of Liver Disease and No Liver Disease Cases")

# Display with bigger font using HTML + CSS
    liver_disease_counts = df['Dataset'].value_counts().reset_index()
    liver_disease_counts.columns = ['Condition', 'Count']
    liver_disease_counts['Condition'] = liver_disease_counts['Condition'].replace({1: 'Liver Disease', 2: 'No Liver Disease'})

    st.markdown("""
    <style>
    .styled-table {
        border-collapse: collapse;
        margin: auto;
        font-size: 20px;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    .styled-table thead tr {
        background-color: #009879;
        color: #ffffff;
    }
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Render table with styling
    st.markdown(
    liver_disease_counts.to_html(index=False, classes='styled-table'),
    unsafe_allow_html=True
)

    image_path = "download.png"
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("""
<div style='text-align: center; color: white; font-size: 30px; font-weight: bold; 
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; 
            padding: 0px 0;'>
    📊 Liver Disease vs No Liver Disease
</div>
""", unsafe_allow_html=True)

        if os.path.exists(image_path):
            st.image(image_path, caption="Liver Disease vs No Liver Disease", width=450)
        else:
            st.error("Error: The image file does not exist.")

# Data Exploratory Tab
with tabs[1]:
    st.header("Data Exploratory")
    st.write("Exploratory Data Analysis results can be displayed here.")
    st.write("Data Summary")
    df_summary = df.describe()
    # Apply custom CSS for styling
    st.markdown("""
<style>
.centered-table {
    border-collapse: collapse;
    margin: auto;
    font-size: 18px;
    font-family: Arial, sans-serif;
    min-width: 600px;
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
    text-align: center;
}
.centered-table thead tr {
    background-color: #2c3e50;
    color: #ffffff;
    text-align: center;
}
.centered-table th,
.centered-table td {
    padding: 4px 4px;
}
</style>
""", unsafe_allow_html=True)

# Render the styled table
    st.markdown(
    df_summary.to_html(classes='centered-table', index=True),
    unsafe_allow_html=True
)

    image_path = "ml.png"
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
         st.markdown("""
<div style='text-align: left; color: white; font-size: 30px; font-weight: bold; 
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; 
            padding: 6px 6;'>
    Correlation </div>
""", unsafe_allow_html=True)       
    if os.path.exists(image_path):
            st.image(image_path, caption="Correlation Heatmap", width=500)
    else:
            st.error("Error: The image file does not exist.")
# Model Performance Tab
with tabs[2]:
    st.header("Model Performance")
    # Load dataset
    df = pd.read_csv("liver.csv")
    # Preprocess
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df.dropna(inplace=True)
    X = df.drop(columns=['Dataset'])
    y = df['Dataset'].replace({1: 0, 2: 1})
    # Apply scaling
    with open("scaler (3).pkl", "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    # Apply SMOTE to match training conditions
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    # Load trained XGBoost model
    with open("xgb_liver_model.pkl", "rb") as f:
        best_model = pickle.load(f)
    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    # Display results
    st.title("Liver Disease Prediction - XGBoost (SMOTE & Tuned)")
    st.subheader("📋 XGBoost Classification Report")

# Generate classification report DataFrame
    report_df = pd.DataFrame(classification_rep).transpose()

# Custom CSS for table
    st.markdown("""
    <style>
    .styled-report-table {
        border-collapse: collapse;
        margin: auto;
        font-size: 18px;
        font-family: Arial, sans-serif;
        min-width: 600px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .styled-report-table thead tr {
        background-color: #4b7bec;
        color: #ffffff;
        text-align: center;
    }
    .styled-report-table th,
    .styled-report-table td {
        padding: 6px 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Render styled classification report table
    st.markdown(
    report_df.to_html(classes='styled-report-table', index=True),
    unsafe_allow_html=True
)
    st.subheader("🎯 Accuracy")

# Custom styled accuracy box
    st.markdown(f"""
    <style>
        .accuracy-box {{
            background-color: #f1f3f6;
            color: #2c3e50;
            font-size: 24px;
            font-weight: bold;
            border-left: 6px solid #27ae60;
            padding: 15px 20px;
            margin: 20px auto;
            width: fit-content;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
    </style>
    <div class="accuracy-box">
        ✅ Accuracy: {accuracy * 100:.2f}%
    </div>
""", unsafe_allow_html=True)
    # Confusion Matrix Image
    image_path1 = "downloadco.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Confusion Matrix")
        if os.path.exists(image_path1):
            st.image(image_path1, caption="Confusion Matrix", width=450)
        else:
            st.error("Error: The image file does not exist.")
            
# Prediction Tab
# Custom CSS with animations and advanced design
st.markdown("""
    <style>
        /* Gradient background for prediction section */

        /* Center the title */
        .css-1v3fvcr {
            text-align: center;
            font-size: 40px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #2c3e50;
            padding-top: 20px;
            animation: fadeIn 1s ease-out;
        }

        /* Button Styling with Hover Animation */
        .stButton>button {
            background-color: #2980b9;
            color: white  !important; 
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 12px 25px;
            width: 250px;
            cursor: pointer;
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1f6391;
            transform: scale(1.05);
            color: white !important;
        }

        /* Input Fields Styling */
        .css-1n0w93d, .css-1n0w93d input {
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ddd;
            margin: 10px 0;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        .css-1n0w93d input:hover {
            background-color: #ecf0f1;
        }

        /* Card-like style for prediction and result */
        .result-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
            animation: fadeIn 1.5s ease-out;
        }

        /* Styled result text */
        .prediction-text {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            color: #2ecc71;  /* Green color for positive outcome */
            margin-top: 20px;
            animation: fadeInUp 1.5s ease-out;
        }

        .prediction-text-no-disease {
            color: #e74c3c;  /* Red color for no disease */
        }

        /* Styled probability text */
        .probability-text {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: #8e44ad;
            margin-top: 10px;
            animation: fadeInUp 2s ease-out;
        }

        /* Fade-in and fade-in-up animations */
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
""", unsafe_allow_html=True)

# Prediction tab content
with tabs[3]:
    # Input Fields in a stylish container
    st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
    
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

    # Prepare input features
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
    
    # End of prediction input container
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction button
    if st.button("Predict"):
        # Use the trained XGBoost model
        prediction = best_model.predict(scaled_features)[0]
        prediction_prob = best_model.predict_proba(scaled_features)[0][1]  # Probability of liver disease
        result = "Liver Disease" if prediction == 1 else "No Liver Disease"
        
        # Display results in a card
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        result_class = "prediction-text" if prediction == 1 else "prediction-text-no-disease"
        st.markdown(f"<div class='{result_class}'>Prediction: {result}</div>", unsafe_allow_html=True)
        
        # Show probability
        st.markdown(f"<div class='probability-text'>Prediction Probability: {prediction_prob:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


