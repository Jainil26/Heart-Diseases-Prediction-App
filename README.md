# ❤️ Heart Disease Prediction App

A beginner-friendly Machine Learning project that predicts whether a person has heart disease based on clinical features. Built using **KNN** and **Random Forest Classifier**, with an interactive **Streamlit** web application deployed on Streamlit Cloud.

---

## 🔗 Live Demo

👉 [Click here to open the app](https://heart-diseases-prediction-app-26.streamlit.app/)

---

## 📌 Project Overview

This project is a **binary classification** problem — the model predicts one of two outcomes:
- `0` → No Heart Disease
- `1` → Heart Disease Detected

The app takes 13 clinical inputs from the user and instantly predicts the result using a trained Random Forest Classifier.

---

## 📂 Project Structure

```
heart-disease-prediction/
│
├── app.py                        # Streamlit web application
├── heart_cleveland_upload.csv    # Dataset (Cleveland Heart Disease)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 📊 Dataset

- **Name:** Cleveland Heart Disease Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Rows:** 303
- **Columns:** 14 (13 input features + 1 target)
- **Target Column:** `condition` (0 = No Disease, 1 = Disease)

### Input Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | Age | Age of the patient |
| 2 | Sex | 0 = Female, 1 = Male |
| 3 | Chest Pain Type | 0 = Typical Angina, 1 = Atypical, 2 = Non-Anginal, 3 = Asymptomatic |
| 4 | Resting Blood Pressure | In mm Hg |
| 5 | Serum Cholesterol | In mg/dl |
| 6 | Fasting Blood Sugar | 1 if > 120 mg/dl, else 0 |
| 7 | Resting ECG Results | 0 = Normal, 1 = ST-T Wave Abnormality, 2 = LV Hypertrophy |
| 8 | Max Heart Rate Achieved | Numeric value |
| 9 | Exercise Induced Angina | 0 = No, 1 = Yes |
| 10 | ST Depression (Oldpeak) | Float value |
| 11 | Slope of Peak Exercise ST | 0 = Upsloping, 1 = Flat, 2 = Downsloping |
| 12 | Number of Major Vessels | 0 to 3 |
| 13 | Thalassemia | 0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect |

---

## 🤖 ML Algorithms Used

### 1. K-Nearest Neighbors (KNN)
- Used the **Elbow Method** to find the best value of K (K=1 to K=20)
- Scaled features using `StandardScaler` before training (required for KNN)

### 2. Random Forest Classifier ⭐ (Final Model)
- Ensemble of 100 Decision Trees
- `n_estimators = 100`, `random_state = 42`
- Better accuracy than KNN on this dataset
- Used for the final Streamlit app

---

## 📈 Model Evaluation

| Metric | KNN | Random Forest |
|--------|-----|---------------|
| Accuracy | ~85% | ~88% |
| Precision | Good | Better |
| Recall | Good | Better |
| F1 Score | Good | Better |

> Random Forest outperformed KNN and was selected as the final model.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | ML algorithms and preprocessing |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Web application framework |

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## ☁️ Deployment

This app is deployed on **Streamlit Cloud**.

Steps followed:
1. Pushed all project files to a public GitHub repository
2. Signed in to [streamlit.io/cloud](https://streamlit.io/cloud) with GitHub
3. Selected the repository and set `app.py` as the main file
4. Clicked Deploy — app went live instantly!

> The model is trained at runtime using `@st.cache_resource` so no `.pkl` files are needed in the repository.

---

## 💡 Key Learnings

- Difference between **Regression** (predicting a number) and **Classification** (predicting a category)
- How **KNN** uses distance to classify and why feature scaling is critical
- How **Random Forest** uses majority voting across many decision trees
- Evaluation metrics for classification: **Accuracy, Confusion Matrix, Precision, Recall, F1 Score**
- Feature importance scores from Random Forest
- Building and deploying an interactive ML web app with Streamlit

---

## 🙋 Author

**Jainil**  
Computer Engineering Student  
[GitHub](https://github.com/Jainil26)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
