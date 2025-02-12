import numpy as np
from matplotlib import pyplot as plt
from model import power_model, determination_coefficient


def stepwise_regression(X_data, Y_data):
    # Перетворення даних (логарифм)
    X_log = np.log(X_data)
    Y_log = np.log(Y_data)

    Mx_2 = np.sum(np.power(X_log,2)) / len(X_log)
    Mx = np.sum(X_log) / len(X_log)
    Mxy = np.sum(X_log * Y_log) / len(X_log)
    My = np.sum(Y_log) / len(X_log)

    m = (Mxy - Mx * My) / (Mx_2 - Mx * Mx)
    b = (Mx_2 * My - Mxy * Mx) / (Mx_2 - Mx * Mx)
    a= np.exp(b)

    y_pred_power = power_model(X_data, m, a)

    equation = f"Степенева регресія :y = {a:.4f} * x^{m:.4f}"


    # Відображення графіків
    plt.scatter(X_data, Y_data, color='red', label='Точки даних')
    plt.plot(X_data, y_pred_power, label='Степенева регресія')
    plt.xlabel('Значення X')
    plt.ylabel('Значення Y')
    plt.title('Аналіз регресії')
    plt.legend()
    plt.show()

    precision = determination_coefficient(Y_data, y_pred_power)
    return equation, precision

