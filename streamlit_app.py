# from collections import namedtuple
# import altair as alt
# import math
# import pandas as pd
# import streamlit as st

# """
# # Welcome to Streamlit!

# Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

# If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
# forums](https://discuss.streamlit.io).

# In the meantime, below is an example of what you can do with just a few lines of code:
# """


# with st.echo(code_location='below'):
#     total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#     num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

#     Point = namedtuple('Point', 'x y')
#     data = []

#     points_per_turn = total_points / num_turns

#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))
import streamlit as st
import numpy as np

# Define the functions to calculate the mean, variance, covariance, and coefficients of a linear regression model.
def mean(values):
  return sum(values) / float(len(values))

def variance(values, mean):
  return sum([(x-mean)**2 for x in values])

def covariance(x, y):
  n = len(x)
  return sum([(x[i]-mean(x))*(y[i]-mean(y)) for i in range(n)]) / float(n-1)

def coefficients(x, y):
  # Calculate the mean and variance of the x and y values.
  mean_x = mean(x)
  mean_y = mean(y)
  var_x = variance(x, mean_x)
  var_y = variance(y, mean_y)

  # Calculate the covariance of the x and y values.
  cov_xy = covariance(x, y)

  # Calculate the coefficients of the linear regression model.
  b1 = cov_xy / var_x
  b0 = mean_y - b1*mean_x

  return b0, b1

# Load the data.
data = np.loadtxt("data.csv", delimiter=",")

# Split the data into training and testing sets.
X_train = data[:, :-1]
y_train = data[:, -1]
X_test = data[:, :-1]
y_test = data[:, -1]

# Calculate the coefficients of the linear regression model.
b0, b1 = coefficients(X_train, y_train)

# Create a Streamlit app.
st.title("Linear Regression")

# Add a slider to select the number of samples.
num_samples = st.slider("Number of samples", 1, 100, 10)

# Generate the data.
x = np.random.randint(0, 100, num_samples)
y = b0 + b1*x + np.random.normal(0, 10, num_samples)

# Plot the data.
st.line_chart(x, y)

# Add a text input to enter the x-value to predict.
x_pred = st.text_input("x-value to predict")

# If the user enters an x-value, predict the y-value.
if x_pred:
  y_pred = b0 + b1*float(x_pred)
  st.write("The predicted y-value is", y_pred)

#     st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#         .mark_circle(color='#0068c9', opacity=0.5)
#         .encode(x='x:Q', y='y:Q'))
