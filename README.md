# 📊 Customer Churn Prediction Machine Learning System

An end-to-end Machine Learning web application designed to predict the likelihood of telecom customer churn based on customer demographics, subscribed services, and account billing attributes.

---

## 📌 Project Overview
Customer retention is crucial for subscription-based businesses. This project applies exploratory data analysis (EDA), trains multiple classification algorithms, identifies the top-performing model, and serves predictions via an intuitive **Streamlit** user interface.

---

## 🛠️ Tech Stack & Tools
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Joblib, Streamlit, Matplotlib, Seaborn
- **Development Environment:** VS Code, Virtual Environment (`.venv`)

---

## 📂 Project Structure
```text
CUSTOMER CHURN PROJECT/
│
├── .venv/
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── model/
│   └── churn_model.pkl
│
├── notebooks/
│   ├── eda.py
│   └── model_training.py
│
├── screenshots/
│   ├── eda/
│   │   └── eda_summary.png
│   ├── model/
│   │   └── model_training.png
│   └── prediction/
│       ├── low_risk.png
│       └── high_risk.png
│
├── main.py
├── requirements.txt
└── README.md