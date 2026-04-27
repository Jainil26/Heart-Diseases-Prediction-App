import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_model():
    df = pd.read_csv('heart_cleveland_upload.csv')
    
    X = df.drop('condition', axis=1)
    y = df['condition']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, scaler

model, scaler = load_model()

st.title("❤️ Heart Disease Prediction App")
st.write("Fill in the details below to check for heart disease risk.")

age      = st.slider("Age", 20, 80, 50)
sex      = st.selectbox("Sex", ["Male", "Female"])
cp       = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol     = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg  = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
thalach  = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang    = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak  = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
slope    = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
ca       = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal     = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Convert inputs to numbers
sex     = 1 if sex == "Male" else 0
cp      = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
fbs     = 1 if fbs == "Yes" else 0
restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
exang   = 1 if exang == "Yes" else 0
slope   = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal    = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected. Please consult a doctor.")
    else:
        st.success("✅ No Heart Disease Detected. Keep up the healthy lifestyle!")