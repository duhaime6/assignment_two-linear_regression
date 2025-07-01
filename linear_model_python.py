#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Check command-line arguments
if len(sys.argv) != 4:
    print("Usage: python linear_regression_combined.py <filename> <x_column> <y_column>")
    sys.exit(1)

# Parse arguments
filename = sys.argv[1]
x_col = sys.argv[2]
y_col = sys.argv[3]

# Load and clean data
try:
    data = pd.read_csv(filename)
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# Check if specified columns exist
if x_col not in data.columns or y_col not in data.columns:
    print(f"Error: Column '{x_col}' or '{y_col}' not found in the file.")
    print(f"Available columns: {', '.join(data.columns)}")
    sys.exit(1)

# Prepare data for modeling
X = data[[x_col]]
y = data[[y_col]]

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Format slope, intercept, and R² for annotation
slope = model.coef_[0][0]
intercept = model.intercept_[0]
r_squared = model.score(X, y)

equation = f'y = {slope:.2f}x + {intercept:.2f}'
r_squared_text = f'$R^2$ = {r_squared:.3f}'  # Use LaTeX for superscript

# Combine into one annotation block
annotation_text = f'{equation}\n{r_squared_text}'

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='red', label='Actual data')
plt.plot(X, predictions, color='blue', label='Regression line')

# Add annotation to the plot
plt.text(
    0.05, 0.95,  # X, Y location in axis coordinates
    annotation_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
)

# Labels and display
plt.title(f'{y_col} vs {x_col}')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.legend()
plt.tight_layout()
plt.savefig("regression_plot_python.png")
plt.show()

# Print regression statistics
slope = model.coef_[0][0]
intercept = model.intercept_[0]
r_squared = model.score(X, y)
mse = mean_squared_error(y, predictions)

print("\nLinear Regression Results:")
print(f"Slope = {slope:.2f}")
print(f"y-intercept = {intercept:.2f}")
print(f"R-squared = {r_squared:.3f}")
print(f"MSE = {mse:.2f}")