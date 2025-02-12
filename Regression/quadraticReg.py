import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols, Eq, solve
from model import quadratic_model, determination_coefficient


def quadratic_regression(X_data, Y_data):

    M_4 = np.sum(np.pow(X_data,4)) / len(X_data)
    M_3 = np.sum(np.power(X_data,3)) / len(Y_data)
    M_2 = np.sum(np.power(X_data,2)) / len(X_data)
    M_1 = np.sum(X_data) / len(X_data)
    Mx_2y = np.sum(np.power(X_data,2 ) *Y_data) / len(X_data)
    Mxy = np.sum(X_data * Y_data) / len(X_data)
    My = np.sum(Y_data) / len(Y_data)



    a, b, c = symbols('a b c')

    # Система рівнянь
    eq1 = Eq(M_4 * a + M_3 * b + M_2 * c, Mx_2y)
    eq2 = Eq(M_3 * a + M_2 + M_1 * c, Mxy)
    eq3 = Eq(M_2 * a + M_1 * b + c, My)

    # Розв'язання системи
    solution = solve((eq1, eq2, eq3), (a, b, c))

    a_value = solution[a]
    b_value = solution[b]
    c_value = solution[c]
    equation =f"Квадратична регресія :{a_value:.4f}*x^2 + {b_value:.4f}*x + {c_value:.4f}"
    y_pred_quadratic = quadratic_model(X_data, a_value, b_value, c_value)

    # Відображення графіків
    plt.scatter(X_data, Y_data, color='red', label='Точки даних')
    plt.plot(X_data, y_pred_quadratic, label='Квадратична регресія')
    plt.xlabel('Значення X')
    plt.ylabel('Значення Y')
    plt.title('Аналіз регресії')
    plt.legend()
    plt.show()

    precision = determination_coefficient(Y_data, y_pred_quadratic)
    return equation, precision