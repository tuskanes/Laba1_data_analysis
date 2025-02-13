import numpy as np
from sklearn.metrics import mean_squared_error


def quadratic_model(x,a,b,c):
    return c + b * x + a * np.power(x,2)

def power_model(x, m, a):
    return a * np.power(x, m)

def exp_model(x, m, a):
    return a * np.exp(x * m)

def hyperbolic_model(x, a, b):
    return (a / x) + b

def log_model(x, a, b):
    return a * np.log(x) + b


def determination_coefficient(y_data, y_pred):
    return round(mean_squared_error(y_data, y_pred),5)
