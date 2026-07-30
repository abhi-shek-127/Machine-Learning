import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =======================================================
# STEP 1: GENERATE DUMMY DATA (e.g., House Size vs Price)
# =======================================================
# X = House size (in 100 sq ft), y = Price (in thousands)
np.random.seed(42)
X = 2 * np.random.rand(50, 1) * 10  # Sizes between 0 and 20
y = 4 + 3 * X + np.random.randn(50, 1) * 2  # Linear relation with added noise

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =======================================================
# STEP 2: IMPLEMENTATION FROM SCRATCH (Mathematical formulas)
# =======================================================
X_mean = np.mean(X_train)
y_mean = np.mean(y_train)

# Calculate Beta 1 (Slope) and Beta 0 (Intercept)
numerator = np.sum((X_train - X_mean) * (y_train - y_mean))
denominator = np.sum((X_train - X_mean) ** 2)

beta_1_scratch = numerator / denominator
beta_0_scratch = y_mean - (beta_1_scratch * X_mean)

# Predict using the custom formulas
y_pred_scratch = beta_0_scratch + beta_1_scratch * X_test

print("--- Implementation From Scratch ---")
print(f"Calculated Intercept (Beta 0): {beta_0_scratch:.4f}")
print(f"Calculated Slope (Beta 1): {beta_1_scratch:.4f}\n")


# =======================================================
# STEP 3: IMPLEMENTATION USING SCIKIT-LEARN
# =======================================================
model = LinearRegression()
model.fit(X_train, y_train)  # Train the model

# Predict using the built-in library function
y_pred_sklearn = model.predict(X_test)

print("--- Implementation via Scikit-Learn ---")
print(f"Sklearn Intercept: {model.intercept_[0]:.4f}")
print(f"Sklearn Coefficient: {model.coef_[0][0]:.4f}\n")


# =======================================================
# STEP 4: EVALUATION METRICS
# =======================================================
print("--- Evaluation (Sklearn Model) ---")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_sklearn):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_sklearn):.4f}")


# =======================================================
# STEP 5: VISUALIZE THE REGRESSION LINE
# =======================================================
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression Blueprint')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.legend()
plt.show()
