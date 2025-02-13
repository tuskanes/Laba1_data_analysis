import numpy as np
from prettytable import PrettyTable
from Regression.linearReg import linear_regression
from Regression.hyperbolicReg import hyperbolic_regression
from Regression.stepwiseReg import stepwise_regression
from Regression.exponentialReg import exponential_regression
from Regression.logarithmicReg import logarithmic_regression
from Regression.quadraticReg import quadratic_regression

# Вхідні дані
#X = np.array([1.73, 2.56, 3.39, 4.22, 5.05, 5.89, 6.7, 7.53])
#Y = np.array([0.63, 1.11, 1.42, 1.96, 2.3, 2.89, 3.29, 3.87])

#REZNIK
X = np.array([1,1.64,2.28,2.91,3.56,4.29,4.84,5.48])
Y = np.array([0.28,0.19,0.15,0.11,0.09,0.08,0.07,0.06])

#X = np.array([5.89,3.84,6.49,9.22,7.87,6.29,4.43,8.91])
#Y= np.array([79.31,57.43,60.66,90.55,92.12,71.30,70.50,91.25])
# Виклик функцій регресії
linreg, line_precision = linear_regression(X, Y)
quadreq, quadratic_precision = quadratic_regression(X, Y)
hyperq, hyper_precision = hyperbolic_regression(X, Y)
stepwiseReg, stepwise_precision = stepwise_regression(X, Y)
expReg, exp_precision = exponential_regression(X, Y)
logreg, log_precision = logarithmic_regression(X, Y)

# Створення таблиці
table = PrettyTable()
table.field_names = ["Метод регресії", "Формула", "Точність"]

# Додавання рядків до таблиці
table.add_row(["Лінійна регресія", linreg, line_precision])
table.add_row(["Квадратична регресія", quadreq, quadratic_precision])
table.add_row(["Гіперболічна регресія", hyperq, hyper_precision])
table.add_row(["Cтепенева регресія", stepwiseReg, stepwise_precision])
table.add_row(["Показникова регресія", expReg, exp_precision])
table.add_row(["Логарифмічна регресія", logreg, log_precision])

# Виведення таблиці
print(table)