# ❤️ Cardiac Health Risk Assessment System

An interactive Machine Learning web application built to assess patient heart disease risk based on clinical parameters. Deployed live using **Streamlit Cloud**.

📌 **Live Web App:** [https://heart-disease-assessment-ai.streamlit.app/](https://heart-disease-assessment-ai.streamlit.app/)

---

## 📌 Features

* **Real-time Prediction**: Calculates the probability of heart disease using a trained Logistic Regression model.
* **Interactive UI**: Form-driven layout built with Streamlit for seamless data entry.
* **Feature Scaling & Encoding**: Automatically handles categorical one-hot encoding and standard scaling (`scaler.pkl`) before inference.
* **Clean & Responsive Interface**: Optimized 2-column input layout with visual risk feedback and no sidebar clutter.

---

## 📊 Dataset & Clinical Parameters

The prediction system evaluates **15 features** covering vital signs, exercise test results, and demographics:

| Feature Category | Description |
| :--- | :--- |
| **Demographics & Vitals** | Age, Sex, Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar |
| **Cardiac Metrics** | Max Heart Rate, Exercise-Induced Angina, Oldpeak (ST Depression) |
| **Categorical Types** | Chest Pain Type (ASY, ATA, NAP, TA), Resting ECG (LVH, Normal, ST), ST Slope (Down, Flat, Up) |
