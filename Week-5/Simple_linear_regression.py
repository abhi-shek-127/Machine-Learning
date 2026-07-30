from sklearn.linear_model import LinearRegression
import numpy as np

# Hours Studied (X)
X = np.array([2, 3, 5, 4, 6]).reshape(-1, 1)

# Marks (Y)
Y = np.array([1, 2, 3, 4, 5])

# Create model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Coefficients
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

# Regression Equation
print("\nRegression Equation:")
print(f"Y = {model.intercept_:.2f} + {model.coef_[0]:.2f}X")

# Predictions
predicted = model.predict(X)

print("\nPredicted Marks:")
for i in range(len(X)):
    print(f"Student {i+1}: Hours={X[i][0]}, Predicted Marks={predicted[i]:.2f}")

# Predict for new student
new_hours = np.array([[8]])
new_marks = model.predict(new_hours)

print("\nPredicted Marks for 8 study hours:", new_marks[0])
