import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("Original Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe(include="all").T)

# ============================================================
# 2. DATA QUALITY CHECK
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nUnique Values:")
print(df.nunique())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# ============================================================
# 3. DATA CLEANING
# ============================================================

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing TotalCharges After Conversion:")
print(df["TotalCharges"].isnull().sum())

# Remove unnecessary customer ID
df = df.drop(columns=["customerID"])

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Remove duplicate rows
df = df.drop_duplicates()

# Convert target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\nDataset Shape After Cleaning:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

print("\nDuplicates After Cleaning:", df.duplicated().sum())

# ============================================================
# 4. EXPLORATORY VISUALIZATION
# ============================================================

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="tenure", bins=30, kde=True)
plt.title("Customer Tenure Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="MonthlyCharges")
plt.title("Monthly Charges Outlier Analysis")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="TotalCharges")
plt.title("Total Charges Outlier Analysis")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()

# ============================================================
# 5. OUTLIER DETECTION USING IQR
# ============================================================

numeric_for_outlier = ["tenure", "MonthlyCharges", "TotalCharges"]

print("\nOutlier Analysis:")

for column in numeric_for_outlier:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(
        column,
        "->",
        len(outliers),
        "potential outliers"
    )

# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

# Tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Short", "Medium", "Long"]
)

# Average monthly value
df["AverageMonthlyValue"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    df["MonthlyCharges"]
)

# Count selected services
service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

service_count = 0

for column in service_columns:
    service_count += (
        df[column]
        .astype(str)
        .isin(["Yes", "Yes"])
        .astype(int)
    )

df["ServiceCount"] = service_count

# High monthly charge indicator
monthly_charge_median = df["MonthlyCharges"].median()

df["HighMonthlyCharge"] = np.where(
    df["MonthlyCharges"] > monthly_charge_median,
    1,
    0
)

print("\nNew Features Created:")
print([
    "TenureGroup",
    "AverageMonthlyValue",
    "ServiceCount",
    "HighMonthlyCharge"
])

print("\nDataset After Feature Engineering:")
print(df.head())

# ============================================================
# 7. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ============================================================
# 9. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

# ============================================================
# 10. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ============================================================
# 11. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ============================================================
# 12. COMPLETE PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer([
    ("numerical", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)

# ============================================================
# 13. FEATURE NAMES AFTER ENCODING
# ============================================================

feature_names = preprocessor.get_feature_names_out()

print("\nNumber of Final Features:", len(feature_names))

print("\nFirst 20 Processed Features:")
print(feature_names[:20])

# ============================================================
# 14. CHECK PROCESSED DATA
# ============================================================

print("\nProcessed Data Check:")

print(
    "Training rows:",
    X_train_processed.shape[0]
)

print(
    "Training features:",
    X_train_processed.shape[1]
)

print(
    "Testing rows:",
    X_test_processed.shape[0]
)

print(
    "Testing features:",
    X_test_processed.shape[1]
)

# ============================================================
# 15. FEATURE SELECTION
# ============================================================

feature_selector = SelectKBest(
    score_func=mutual_info_classif,
    k=min(20, X_train_processed.shape[1])
)

X_train_selected = feature_selector.fit_transform(
    X_train_processed,
    y_train
)

X_test_selected = feature_selector.transform(
    X_test_processed
)

selected_features = feature_names[
    feature_selector.get_support()
]

print("\nSelected Feature Count:", len(selected_features))

print("\nSelected Features:")
print(selected_features)

print(
    "\nSelected Training Shape:",
    X_train_selected.shape
)

print(
    "Selected Testing Shape:",
    X_test_selected.shape
)

# ============================================================
# 16. OPTIONAL PCA
# ============================================================

pca = PCA(
    n_components=0.95,
    random_state=42
)

X_train_pca = pca.fit_transform(
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

X_test_pca = pca.transform(
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed
)

print("\nPCA Components:", pca.n_components_)

print(
    "PCA Training Shape:",
    X_train_pca.shape
)

print(
    "PCA Testing Shape:",
    X_test_pca.shape
)

print(
    "\nExplained Variance:",
    round(pca.explained_variance_ratio_.sum(), 4)
)

# ============================================================
# 17. FINAL VALIDATION
# ============================================================

print("\n================ FINAL VALIDATION ================")

print(
    "Missing Values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate Rows:",
    df.duplicated().sum()
)

print(
    "Original Shape:",
    df.shape
)

print(
    "Processed Training Shape:",
    X_train_processed.shape
)

print(
    "Processed Testing Shape:",
    X_test_processed.shape
)

print(
    "Selected Feature Shape:",
    X_train_selected.shape
)

print(
    "PCA Shape:",
    X_train_pca.shape
)

print("\nWeek 2 Data Preprocessing and Feature Engineering Completed Successfully!")
