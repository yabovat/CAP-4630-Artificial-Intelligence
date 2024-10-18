# -*- coding: utf-8 -*-
"""Homework_01.ipynb

# Regression Homework
This is the first assignment for CAP 4630 and we will develop two basic models with regression. \
You will use **"Tasks"** and **"Hints"** to finish the work. **(Total 100 Points)**

Summary:
This homework involves developing two regression models as part of the first assignment for CAP 4630. The key tasks are to implement:

1. Single Variable Nonlinear Regression:
   - Established a cubic function to model the relationship between variables X and Y.
   - Used Mean Squared Error (MSE) to evaluate the model's performance.
   - Applied Gradient Descent to optimize the model coefficients a, b, c, and d over 10,000 epochs with a learning rate of 1e-6
   - The optimal coefficients were printed at the final epoch, and a prediction function was derived using these coefficients.

2. Multiple Variable Linear Regression:
   - Loaded data with two independent variables X_1 and X_2, and the dependent variable Y.
   - Visualized the data with a 3D scatter plot.
   - Established a linear function and used Gradient Descent to optimize the coefficients m_1, m_2, and m_3.
   - Similar to the single-variable model, MSE was used to compute the loss, and the results were printed at each epoch for debugging.

Both models utilized common data science libraries like NumPy, Pandas, and Matplotlib for data processing, visualization, and mathematical computations.


**Task Overview:**
- Singal Variable Nonlinear Regression
- Multiple Variable Linear Regression

Name: Yahya Abovat

Collaboration: Mujahid Khan

## 1 - Packages ##

Import useful packages for scientific computing and data processing. **(5 Points)**

**Tasks:**
1. Import numpy and rename it to np.
2. Import pandas and rename it to pd.
3. Import the pyplot function in the libraray of matplotlib and rename it to plt.

References:
- [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
- [matplotlib](http://matplotlib.org) is a famous library to plot graphs in Python.

**Attention:**
1. After this renaming, you will use the new name to call functions. For example, **numpy** will become **np** in the following sections.
"""

# Task 1: Import Statments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""## 2 - Data Preparation ##

Prepare the data for regression task. **(10 Points)**

**Tasks:**
1. Load data for nonlinear regression.
2. Generate the scatter plot of the data.

**Hints:**
1. The data file is "data_nonlinear.csv".
2. The data format is as follows: 1st column is X and 2nd column is Y.
"""

from google.colab import files
uploaded = files.upload()

for filename, content in uploaded.items():
    print(f'Uploaded file "{filename}" with {len(content)} bytes')

# Load data for nonlinear regression
data_path = "data_nonlinear.csv"
df = pd.read_csv(data_path)

# Display the first few rows of the dataset
print(df.head())

# Generate the scatter plot
plt.scatter(df['X'], df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Nonlinear Regression Data')
plt.show()

"""## 3 - Single Variable Nonlinear Regression ##


Develop a regression model, estimate coefficients with data, and derive the relationship. **(30 Points)**

**Tasks:**
1. Establish a relationship between Y and X with a cubic function.
2. Compute MSE loss with observation-prediction pairs.
3. Implement **Gradient Descent (GD)** to achieve optimal solution with the learning rate of **0.000001 (1e-6)** and **10000 (1e4)** epochs.
4. Print out the optimal solution at final step.

**Hints:**
1. Given the example of linear regression in class, modify the function to an equation for a spline with coefficients of **a** , **b**, **c** and **d** for cubic, qudractic, linear, and constant term.
2. Initialize the model with zero. For example, a=0, b=0, c=0 and d=0.
3. It may take **10-15 seconds**  to finish the running for 10000 steps. Be patient.
4. For debugging, the results of **a**, **b**, **c**, **d** for first five steps are as follows:

Epoch  0 :  2.8045093168662314 0.15006631239563697 0.04047903434004733 0.0030023401200892003 \
Epoch  1 :  4.905935374329749 0.2803623842843468 0.07068280026181122 0.0057565282228493 \
Epoch  2 :  6.480417434500056 0.395779237410925 0.09318576969022647 0.008323648642107889 \
Epoch  3 :  7.65996806232127 0.49998280146312246 0.10991745268097952 0.010749486523089888 \
Epoch  4 :  8.543527816733905 0.5957208253596222 0.12232397430880633 0.013068360586717544


"""

X_data = df['X'].values
Y_data = df['Y'].values

# Task 1: Establish a relationship between Y and X with a cubic function
def cubic_spline_function(X, a, b, c, d):
    return a * X**3 + b * X**2 + c * X + d

# Task 2: Compute MSE loss with observation-prediction pairs
def compute_mse_loss(Y, Y_pred):
    return np.mean((Y - Y_pred)**2)

# Task 3: Implement Gradient Descent
def gradient_descent(X, Y, learning_rate, epochs):
    m = len(X)
    a, b, c, d = 0, 0, 0, 0  # Initialize coefficients

    for epoch in range(epochs):
        Y_pred = cubic_spline_function(X, a, b, c, d)

        # Calculate gradients (partial derivatives) with respect to each coefficient
        gradient_a = (-2/m) * np.sum((Y - Y_pred) * X**3)
        gradient_b = (-2/m) * np.sum((Y - Y_pred) * X**2)
        gradient_c = (-2/m) * np.sum((Y - Y_pred) * X)
        gradient_d = (-2/m) * np.sum(Y - Y_pred)

        # Update coefficients using the gradients and learning rate
        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b
        c -= learning_rate * gradient_c
        d -= learning_rate * gradient_d

        if epoch < 5:
            print(f'Epoch {epoch}: a={a}, b={b}, c={c}, d={d}')

    # Print out the optimal solution at the final step
    print(f'Final Epoch: a={a}, b={b}, c={c}, d={d}')
    return a, b, c, d

learning_rate = 0.000001
epochs = 10000

# Perform gradient descent
optimal_solution = gradient_descent(X_data, Y_data, learning_rate, epochs)

"""## 4 - Prediction Results ##

Derive prediction function and generate estmated results. **(5 Points)**

**Tasks:**
1. Derive prediction function with the obtained coefficients above.
2. Generate scatter plots for original data pairs X-Y and prediction results X-Y_Pred in the same figure.
"""

def prediction_function(X):
    return a_coefficient * X**3 + b_coefficient * X**2 + c_coefficient * X + d_coefficient

X_to_predict = 10
predicted_Y = prediction_function(X_to_predict)
print(f'For X={X_to_predict}, predicted Y is {predicted_Y}')

# Generate prediction results (Y_Pred)
Y_Pred = prediction_function(X_data)

# Create scatter plots
plt.scatter(X_data, Y_data, label='Original Data (X-Y)', color='green')
plt.scatter(X_data, Y_Pred, label='Prediction Results (X-Y_Pred)', color='red', marker='x')

# Add labels and legend and print
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

"""## 5 - Multiple Variables Linear Regression ##

## 5.1 Data Preparation

Prepare the data for regression task. **(10 Points)**

**Tasks:**
1. Load data for multiple variable linear regression.
2. Generate the 3D scatter plot of the data.

**Hints:**
1. The data file is "data_two_variables.csv".
2. The data format is as follows: 1st column is X1, 2nd column is X2, and 3rd colum is Y.
3. You may use "mplot3d" in the toolkit of "mpl_toolkits" and import "Axes3D" to faciliate 3D scatter plot. More details can be found in the reference of https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
4. [Optional, NO Credit]You may rotate the figure you generated by using "%matplotlib qt" before you plot it. Remember to install the related package by "!pip install PyQt5". Only work on Jupyter(locally). Does not work on Google Colab. [Reference Website](https://stackoverflow.com/questions/14261903/how-can-i-open-the-interactive-matplotlib-window-in-ipython-notebook)

![](https://drive.google.com/uc?export=view&id=1sHwWfZXpU3-8SqzFrmCxIvxmQWfe2Nns)
![](https://drive.google.com/uc?export=view&id=1OwHP0g-K2um-LnKiDhE6UfkDFxk4Opce)

"""

from google.colab import files
uploaded = files.upload()

for filename, content in uploaded.items():
    print(f'Uploaded file "{filename}" with {len(content)} bytes')

data_path = "data_two_variables.csv"
df = pd.read_csv(data_path)

# Display the first few rows of the data
print("First few rows of the data:")
print(df.head())

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make a scatter plot with X1, X2, and Y columns
ax.scatter(df['X1'], df['X2'], df['Y'], c='blue', marker='o')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

plt.show() # plot

"""
## 5.2 Linear Regression

Develop a regression model, estimate coefficients with data, and derive the relationship. **(30 Points)**

**Tasks:**
1. Establish a linear function to describe the relationship among Y, X1, and X2.
2. Compute MSE loss with observation-prediction pairs.
3. Implement **Gradient Descent (GD)** to achieve optimal solution with the learning rate of **0.001 (1e-3)** and **10000 (1e4)** epochs.
4. Print out the optimal solution at final step.


**Hints:**
1. Given the example of linear regression in class, modify the function to a linear equation with two independent variables X1 and X2. The coefficients of X1 and X2 are **m1** and **m2**, respectively. The constant term is **m3**.
2. Initialize the model with zero. For example, m1=0, m2=0, and m3=0.
3. It may take **10-15 seconds**  to finish the running for 10000 steps. Be patient.
4. For debugging, the results of **m1**, **m2**, and **m3** for first five steps are as follows:

Epoch 0: 7.43847600018326 15.595631430047339 1.4265844915879997 \
Epoch 1: 12.954483113402425 26.731746959534096 2.481143659135288 \
Epoch 2: 17.084193849045587 34.664109745712814 3.2680146970514863 \
Epoch 3: 20.213137348549306 40.2953527521597 3.8622050343066556 \
Epoch 4: 22.618552798604984 44.274269323103674 4.317638791453634 \
Epoch 5: 32.734943422646175 34.69592128962032 222.91661391579638"""

def gradient_descent(X1, X2, Y, m1, m2, m3, learning_rate, epochs):
    global m1_new, m2_new, m3_new  # Global variables

    m = len(X1)  # Number of data points

    for epoch in range(epochs):
        # Use the linear function to make predictions
        predictions = linear_function(X1, X2, m1, m2, m3)

        # Calculate the gradient (partial derivatives)
        gradient_m1 = (-2/m) * np.sum((Y - predictions) * X1)
        gradient_m2 = (-2/m) * np.sum((Y - predictions) * X2)
        gradient_m3 = (-2/m) * np.sum(Y - predictions)
        m1 -= learning_rate * gradient_m1
        m2 -= learning_rate * gradient_m2
        m3 -= learning_rate * gradient_m3

        # For debugging, print the magic numbers for the first five steps
        if epoch < 5:
            print(f'Epoch {epoch}: m1={m1}, m2={m2}, m3={m3}')

    # Print out the optimal solution at the final step
    print(f'Final Epoch: m1={m1}, m2={m2}, m3={m3}')

    m1_new = m1
    m2_new = m2
    m3_new = m3

# Perform Gradient Descent
gradient_descent(df['X1'], df['X2'], df['Y'], m1, m2, m3, learning_rate, epochs)

# Set the learning rate and number of epochs
learning_rate = 0.001
epochs = 10000

"""
## 5.3 - Prediction Results ##

Derive prediction function and generate estmated results. **(10 Points)**

**Tasks:**
1. Derive prediction function with the obtained coefficients above.
2. Generate 3D scatter plots for original data pairs X-Y and prediction results X-Y_Pred in the same figure.

**Hint:**
1. You may follow the example above.
2. An example is shown below.
![](https://drive.google.com/uc?export=view&id=1xAl7eJmDmFPTNipd0SljAdyHs3PhRiMg)
![](https://drive.google.com/uc?export=view&id=1Eb9qZqTCmAbwJUkoTQ6zPys3ezWqTCkr)"""

# Task 1: Derive the prediction function with the obtained coefficients
def prediction_function(X1, X2, m1_new, m2_new, m3_new):
    return m1_new * X1 + m2_new * X2 + m3_new
Y_Pred = prediction_function(df['X1'], df['X2'], m1_new, m2_new, m3_new)

# Generate 3D scatter plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for original data pairs (X1, X2, Y)
ax.scatter(df['X1'], df['X2'], df['Y'], label='Original Data (X1, X2, Y)', color='blue', marker='o')

# Scatter plot for prediction results (X1, X2, Y_Pred)
ax.scatter(df['X1'], df['X2'], Y_Pred, label='Prediction Results (X1, X2, Y_Pred)', color='red', marker='o')

# Set labels for each axis and plot graph
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.show()
