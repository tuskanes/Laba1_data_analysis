import numpy as np
from matplotlib import pyplot as plt
from model import log_model, determination_coefficient


def logarithmic_regression(X_data, Y_data):
    X_log = np.array(np.log(X_data))

    # Визначення коефіцієнтів
    Mx_2 = np.sum(np.power(X_log,2)) / len(X_log)
    Mx = np.sum(X_log) / len(X_log)
    Mxy = np.sum(X_log * Y_data) / len(X_log)
    My = np.sum(Y_data) / len(Y_data)

    a = (Mxy - Mx * My) / (Mx_2 - Mx * Mx)
    b = (Mx_2 * My - Mxy * Mx) / (Mx_2 - Mx * Mx)

    equation = f"Логарифмічна регресія: y ={a:.4f} * ln(x) + {b:.4f}"

    y_pred_log = log_model(X_data, a, b)

    # Відображення графіків
    plt.scatter(X_data, Y_data, color='red', label='Точки даних')
    plt.plot(X_data, y_pred_log, label='Логарифмічна регресія')
    plt.xlabel('Значення X')
    plt.ylabel('Значення Y')
    plt.title('Аналіз регресії')
    plt.legend()
    plt.show()

    precision = determination_coefficient(Y_data, y_pred_log)
    return equation, precision