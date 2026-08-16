import os
import glob
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset auto-detect
csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"Koi CSV file nahi mili dataset folder mein: {DATASET_DIR}")

dataset_path = csv_files[0]
print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

# 2. DATA PREPROCESSING
# Drop customerID agar maujood ho
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# TotalCharges handle karein (spaces to NaN and fill with median)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_val = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_val)

# SeniorCitizen mapping check (ensure int)
if "SeniorCitizen" in df.columns:
    df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

# Target Column convert to binary (0 and 1)
if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)

# Categorical columns encoding
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Confirm no missing values remain
df = df.fillna(0)

# 3. SPLIT FEATURES & TARGET
X = df.drop(columns=["Churn"])
y = df["Churn"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. MODEL INITIALIZATION & TRAINING
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []
trained_models = {}

print("\n" + "=" * 50)
print("TRAINING AND EVALUATING MODELS")
print("=" * 50)

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(auc, 4)
    })
    trained_models[name] = model

# 5. MODEL COMPARISON RESULTS
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print("\n", results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBEST MODEL SELECTED: {best_model_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']})")

# 6. SAVE ARTIFACTS
save_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "scaler": scaler,
    "label_encoders": label_encoders,
    "feature_names": feature_names
}

model_save_path = os.path.join(MODEL_DIR, "churn_model.pkl")
joblib.dump(save_bundle, model_save_path)

print("\n" + "=" * 50)
print(f"Model and Preprocessing objects saved to: {model_save_path}")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)