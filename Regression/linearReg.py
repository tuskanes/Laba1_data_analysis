import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from model import determination_coefficient
# Введення даних
def linear_regression(X_data, Y_data):
    # Лінійна регресія
    X_data= np.array(X_data).reshape(-1, 1)
    model_linear = LinearRegression()
    model_linear.fit(X_data, Y_data)
    y_pred_linear = model_linear.predict(X_data)

    #Визначення коефіцієнтів
    Mx_2 = np.sum(X_data**2) / len(X_data)
    Mx = np.sum(X_data) / len(X_data)
    Mxy = np.sum(X_data * Y_data) / len(X_data)
    My = np.sum(Y_data) / len(Y_data)

    a = (Mxy - Mx * My)/(Mx_2 - Mx*Mx)
    b = (Mx_2*My - Mxy*Mx)/(Mx_2 - Mx*Mx)
    equation = f"y = {a:.4f}x + {b:.4f}"
    # Відображення графіків
    plt.scatter(X_data, Y_data, color='red', label='Точки даних')
    plt.plot(X_data, y_pred_linear, label='Лінійна регресія')
    plt.xlabel('Значення X')
    plt.ylabel('Значення Y')
    plt.title('Аналіз регресії')
    plt.legend()
    plt.show()

    precision =determination_coefficient(Y_data, y_pred_linear)
    return equation, precision