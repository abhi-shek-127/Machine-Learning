## from sklearn.linear_model import LinearRegression
import numpy as np

# Features: [Study Hours, Attendance] (3 students)
X = np.array([[2, 60], [4, 70], [5, 75]])

# Marks (3 students)
Y = np.array([39, 48, 52])

# Create model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Coefficients
print("Intercept (b0):", model.intercept_)
print("Coefficient for Study Hours (b1):", model.coef_[0])
print("Coefficient for Attendance (b2):", model.coef_[1])

# Predict Student D
student_A =np.array([[2,60]])
prediction = model.predict(student_A)
print("\nPredicted Marks for Student A:", prediction[0])
student_B =np.array([[4,70]])
prediction = model.predict(student_B)
print("\nPredicted Marks for Student B:", prediction[0])
student_C =np.array([[5,75]])
prediction = model.predict(student_C)
print("\nPredicted Marks for Student C:", prediction[0])
student_D = np.array([[6, 80]])
prediction = model.predict(student_D)
print("\nPredicted Marks for Student D:", prediction[0])
