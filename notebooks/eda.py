import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# CUSTOMER CHURN PROJECT - EDA
# ==========================================

print("=" * 60)
print("CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

file_path = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")


# ==========================================
# STEP 2: DATA CLEANING
# ==========================================

# Convert TotalCharges from string to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Remove customerID
df = df.drop(columns=["customerID"])

# Remove duplicate rows
df = df.drop_duplicates()


# ==========================================
# STEP 3: DATA INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("CLEANED DATASET INFORMATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==========================================
# STEP 4: CHURN DISTRIBUTION
# ==========================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(
    df["Churn"].value_counts(normalize=True) * 100
)


# ==========================================
# GRAPH 1: CHURN DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 2: GENDER VS CHURN
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="gender",
    hue="Churn"
)

plt.title("Gender vs Customer Churn")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 3: SENIOR CITIZEN VS CHURN
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="SeniorCitizen",
    hue="Churn"
)

plt.title("Senior Citizen vs Customer Churn")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 4: TENURE VS CHURN
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 5: MONTHLY CHARGES VS CHURN
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 6: CONTRACT VS CHURN
# ==========================================

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 7: INTERNET SERVICE VS CHURN
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Internet Service vs Customer Churn")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 8: PAYMENT METHOD VS CHURN
# ==========================================

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Payment Method vs Customer Churn")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.xticks(rotation=25)

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 9: TECH SUPPORT VS CHURN
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="TechSupport",
    hue="Churn"
)

plt.title("Tech Support vs Customer Churn")
plt.xlabel("Tech Support")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 10: ONLINE SECURITY VS CHURN
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OnlineSecurity",
    hue="Churn"
)

plt.title("Online Security vs Customer Churn")
plt.xlabel("Online Security")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# STEP 5: BUSINESS ANALYSIS
# ==========================================

print("\n" + "=" * 60)
print("BUSINESS ANALYSIS")
print("=" * 60)


# Churn rate by Contract
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Contract (%):")
print(contract_churn.round(2))


# Churn rate by Internet Service
internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Internet Service (%):")
print(internet_churn.round(2))


# Churn rate by Payment Method
payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Payment Method (%):")
print(payment_churn.round(2))


# Churn rate by Tech Support
support_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Tech Support (%):")
print(support_churn.round(2))


# ==========================================
# PROJECT COMPLETED
# ==========================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)