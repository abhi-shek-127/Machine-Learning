import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
# (Note: Fetching from openml since it was deprecated in scikit-learn)
from sklearn.datasets import fetch_openml
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# Step 1: Convert the continuous target (MEDV) into binary classes
# 1 = Above median price (High value), 0 = Below/equal to median price (Low value)
median_price = df['MEDV'].median()
df['PRICE_CLASS'] = (df['MEDV'] > median_price).astype(int)

# Define Features (X) and Binary Target (y)
X = df.drop(columns=['MEDV', 'PRICE_CLASS'])
y = df['PRICE_CLASS']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale features (Crucial for Logistic Regression convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 5: Make Predictions and Evaluate
y_pred = model.predict(X_test_scaled)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
