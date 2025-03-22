import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from euler import euler_method

# Parámetros del circuito RC
R = 1000.0  # Resistencia en ohmios
C = 0.0010  # Capacitancia en faradios
V_in = 10.0  # Voltaje de entrada constante

# Definir la ecuación diferencial dy/dx = (1/RC) * (V_in - y)
def dydx(x, y):
    """
    Calcula la derivada dy/dx para la ecuación del circuito RC.

    Args:
        x (float): El tiempo (s).
        y (float): El voltaje en el capacitor (V).

    Returns:
        float: La derivada dy/dx.

    Examples:
        >>> dydx(0, 0)
        10.0
        >>> dydx(1, 5)
        5.0
    """
    return (1 / (R * C)) * (V_in - y)

# Solución exacta de la ecuación diferencial para comparación
def exact_solution(x):
    """
    Calcula la solución exacta para la ecuación del circuito RC.

    Args:
        x (numpy.ndarray or float): El tiempo (s).

    Returns:
        numpy.ndarray or float: El voltaje en el capacitor (V).

    Examples:
        >>> np.allclose(exact_solution(0), 0)
        True
        >>> np.allclose(exact_solution(1), 6.321205588285577)
        True
    """
    return V_in * (1 - np.exp(-x / (R * C)))

# Condiciones iniciales
x0 = 0
y0 = 0

# Tamaño del paso
h = 0.01

# Intervalo de integración
x_end = 5

# Calcular el número de pasos usando la fórmula de número de pasos
n = int((x_end - x0) / h) + 1

# Resolver la ecuación diferencial usando el método de Euler
x_euler, y_euler = euler_method(dydx, x0, y0, h, n)

# Calcular la solución exacta
y_exacta = exact_solution(x_euler)

# Guardar los resultados exactos en un diccionario
resultados_exactos = {'x': x_euler.tolist(), 'y': y_exacta.tolist()}    

# Graficar los resultados
plt.figure()
plt.plot(x_euler, y_euler, 'b-', label='Método de Euler')
plt.plot(x_euler, y_exacta, 'r--', label='Solución Exacta')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.legend()
plt.title('Comparación del Método de Euler con la Solución Exacta para la Ecuación del Circuito RC')
plt.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod()