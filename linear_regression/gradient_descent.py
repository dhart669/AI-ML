
### `linear_regression/gradient_descent.py`

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data from CSV file
file_path = '../data/linear_regression_data.csv'
data = pd.read_csv(file_path)

# Extract Size (sq ft) and Price ($) columns
X = data['Size (sq ft)'].values
y = data['Price ($)'].values

# Normalize the data
X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)

# Parameters
learning_rate = 0.01
num_iterations = 1000
m = len(y)

# Initialize weights and bias
w = 0.0
b = 0.0

# Gradient Descent
for i in range(num_iterations):
    # Predicted values
    y_pred = w * X + b
    
    # Compute the cost (MSE)
    cost = (1/m) * np.sum((y_pred - y)**2)
    
    # Compute gradients
    dw = (2/m) * np.sum((y_pred - y) * X)
    db = (2/m) * np.sum(y_pred - y)
    
    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Print cost every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Cost {cost}")

# Plotting the results
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, w * X + b, color='red', label='Fitted Line')
plt.xlabel('Size (normalized)')
plt.ylabel('Price (normalized)')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()

# Final parameters
print("Final weight (w):", w)
print("Final bias (b):", b)

