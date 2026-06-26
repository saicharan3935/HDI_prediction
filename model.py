import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Read Dataset
data = pd.read_csv("HDI.csv", encoding="latin1")

# Display first 5 rows
print(data.head())

# Dataset information
print(data.info())

# Shape
print(data.shape)

# Missing values
print(data.isnull().sum())

# Fill missing values
data = data.ffill()

# Convert GNI column to numeric
data["Gross national income (GNI) per capita"] = (
    data["Gross national income (GNI) per capita"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

data["Gross national income (GNI) per capita"] = pd.to_numeric(
    data["Gross national income (GNI) per capita"],
    errors="coerce"
)

# Remove remaining missing values
data.dropna(inplace=True)

# Encode target column
le = LabelEncoder()
data["HUMAN DEVELOPMENT"] = le.fit_transform(data["HUMAN DEVELOPMENT"])

print("Preprocessing Completed!")
print(data.head())
print(data.isnull().sum())
# Features and Target
X = data.drop("HUMAN DEVELOPMENT", axis=1)
y = data["HUMAN DEVELOPMENT"]

# Convert Country to numbers
le_country = LabelEncoder()
X["Country"] = le_country.fit_transform(X["Country"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# Train Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save Model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")