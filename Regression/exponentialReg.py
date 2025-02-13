import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from model import exp_model, determination_coefficient


def exponential_regression(X_data, Y_data):
    Y_log = np.log(Y_data)

    # Визначення коефіцієнтів
    Mx_2 = np.sum(X_data ** 2) / len(X_data)
    Mx = np.sum(X_data) / len(X_data)
    Mxy = np.sum(X_data * Y_log) / len(X_data)
    My = np.sum(Y_log) / len(Y_log)

    m = (Mxy - Mx * My) / (Mx_2 - Mx * Mx)
    b = (Mx_2 * My - Mxy * Mx) / (Mx_2 - Mx * Mx)
    a = np.exp(b)
    equation = f"y = {a:.4f} * e^{m:.4f}x"

    y_pred_exp = exp_model(X_data, m, a)

    # Відображення графіків
    plt.scatter(X_data, Y_data, color='red', label='Точки даних')
    plt.plot(X_data, y_pred_exp, label='Показникова регресія')
    plt.xlabel('Значення X')
    plt.ylabel('Значення Y')
    plt.title('Аналіз регресії')
    plt.legend()
    plt.show()

    precision = determination_coefficient(Y_data, y_pred_exp)
    return equation, precision
