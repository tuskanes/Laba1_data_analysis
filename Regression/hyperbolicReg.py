import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error
from model import hyperbolic_model, determination_coefficient


def hyperbolic_regression(X_data, Y_data):
    u_data = np.array([1 / x for x in X_data])
    u_2 = np.array([np.power(u, 2) for u in u_data])
    uy = np.array([u * y for u, y in zip(u_data, Y_data)])

    Mu = np.sum(u_data ) /len(X_data)
    My = np.sum(Y_data ) /len(X_data)
    Mu_2 = np.sum(u_2 ) /len(X_data)
    Muy = np.sum(uy ) /len(X_data)


    a = (Muy -Mu *My) / (Mu_2 -Mu*Mu)
    b = (Mu_2 *My - Mu* Muy) / (Mu_2 -Mu*Mu)
    equation = f"Гіперболічна регресія :{a:.4f}/x + {b:.4f}"

    y_pred_hyper = hyperbolic_model(X_data, a, b)

    # Відображення графіків
    plt.scatter(X_data, Y_data, color='red', label='Точки даних')
    plt.plot(X_data, y_pred_hyper, label='Гіперболічна регресія')
    plt.xlabel('Значення X')
    plt.ylabel('Значення Y')
    plt.title('Аналіз регресії')
    plt.legend()
    plt.show()

    precision = determination_coefficient(Y_data, y_pred_hyper)

    return equation, precision