# Heart Disease Prediction (ML + Flask)

This project is a diploma project completed as part of the **Proweb Data Science program**.

It focuses on predicting the risk of cardiovascular disease using machine learning models and provides a web interface for real-time predictions.

---

## 📊 Project Overview

The goal of this project is to build a machine learning model that predicts whether a patient has a risk of cardiovascular disease based on medical and lifestyle features.

The project includes:
- Data analysis and preprocessing
- Model training and comparison
- Model selection and evaluation
- Deployment via a Flask web application

---

## 🗂 Dataset

- Source: Public dataset (Kaggle / open sources)
- Records: ~70,000 patients
- Features: 12 input features + 1 target

### Target Variable
- `cardio`
  - 0 — no disease
  - 1 — cardiovascular disease present

### Features include:
- Age (in days)
- Gender
- Height / Weight
- Blood pressure (ap_hi, ap_lo)
- Cholesterol
- Glucose
- Smoking
- Alcohol consumption
- Physical activity

---

## ⚙️ Data Analysis & Processing

- Data cleaning and validation
- Feature understanding and visualization
- BMI (Body Mass Index) analysis
- Checking class balance (dataset is balanced ~50/50)
- No missing values

---

## 🤖 Machine Learning Models

Several models were tested:

- Random Forest
- K-Nearest Neighbors (K=5, K=9)
- Logistic Regression ✅ (selected)

### Why Logistic Regression?

- Designed for binary classification
- Works well with medical data
- Less prone to overfitting
- Fast training and prediction
- Provides probability output (important for risk estimation)

---

## 📈 Model Objective

Predict the probability of cardiovascular disease based on patient data and classify:

- Healthy
- At Risk

---

## 🌐 Web Application (Flask)

A web application was built using Flask to make predictions accessible to users.

### Features:
- Input form for patient data
- Real-time prediction using trained ML model
- Clean UI
- Multi-language support:
  - English
  - Russian
  - Uzbek

### How it works:
1. User inputs medical data
2. Data is processed and passed to the model
3. Model returns prediction
4. Result is displayed instantly

---

## 🛠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook
- PyCharm
- Flask

---

## 🎯 Project Purpose

- Demonstrate ML workflow from data analysis to deployment
- Solve a real-world healthcare problem
- Build an interactive prediction system

---

## 📌 Notes

- This is a diploma project from **Proweb Data Science**
- The model is for educational purposes and not for medical use
