import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

if len(sys.argv) != 4:
    print("Usage: python linear_regression_python.py <filename> <x_column> <y_column>")
    sys.exit(1)

filename = sys.argv[1]
x_col = sys.argv[2]
y_col = sys.argv[3]

data = pd.read_csv(filename)
model = LinearRegression()
model.fit(data[[x_col]], data[[y_col]])

plt.scatter(data[[x_col]], data[[y_col]], color='red')
plt.plot(data[[x_col]], model.predict(data[[x_col]]), color='blue')
plt.title(f'{y_col} vs {x_col}')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.savefig("linear_regression_python_output.png")
plt.show()

#!/usr/bin/env python
# coding: utf-8

# # Linear Regression in Python
# This notebook demonstrates a simple linear regression analysis using Python to model Salary based on Years of Experience.

# In[1]:


import pandas as pd


# Imports the pandas library and gives it a short name (pd). Pandas is a powerful Python library for working with tabular data (like spreadsheets or CSV files).

# In[2]:


dataset = pd.read_csv("regression_data.csv")


# Reads a CSV file and loads it into a DataFrame, which is like a table in memory that you can work with in Python.

# In[3]:


import matplotlib.pyplot as plt


# Loads the matplotlib plotting library, which is used to create charts and visualizations.

# In[4]:


plt.scatter(dataset["YearsExperience"], dataset["Salary"], color="red")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(dataset[["YearsExperience"]], dataset[["Salary"]])
plt.plot(dataset["YearsExperience"], model.predict(dataset[["YearsExperience"]]), color="blue")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# **plt.scatter(x, y) :** Creates a scatter plot — red dots on a graph showing the original data points.
# 
# **from sklearn.linear_model import LinearRegression :** Imports the LinearRegression class from the scikit-learn (sklearn) library. Scikit-learn is a widely used machine learning library.
# 
# **model.fit(x, y) :** Fits the linear regression model to the data. X is the input (independent variable, must be 2D), y is the target (dependent variable).
# 
# **plt.plot(x, y_pred) :** Draws the regression line using predicted values. The blue line is the best-fit line the model calculates.

# In[5]:


correlation_coefficient = model.score(dataset[["YearsExperience"]], dataset[["Salary"]])  # R-squared


# In[6]:


print(f"Correlation coefficient = {correlation_coefficient:.3f}")

