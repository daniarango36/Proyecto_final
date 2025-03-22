import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from euler import euler_method2

# Parámetro de la ecuación
u = 2.0

# Definir la ecuación diferencial dx/dt = y, dy/dt = u(1-x^2)y - x
def dxdt(t, x, y):
    """
    Calcula la derivada dx/dt.

    Args:
        t (float): El tiempo.
        x (float): El valor de x.
        y (float): El valor de y.

    Returns:
        float: La derivada dx/dt.
    """
    return y

def dydt(t, x, y):
    """
    Calcula la derivada dy/dt.

    Args:
        t (float): El tiempo.
        x (float): El valor de x.
        y (float): El valor de y.

    Returns:
        float: La derivada dy/dt.
    """
    return u * (1 - x**2) * y - x

# Definir el sistema de ecuaciones diferenciales como una función
def system(t, z):
    """
    Define el sistema de ecuaciones diferenciales.

    Args:
        t (float): El tiempo.
        z (list): Una lista que contiene los valores de x e y.

    Returns:
        list: Una lista que contiene las derivadas dx/dt y dy/dt.
    """
    x, y = z
    return [dxdt(t, x, y), dydt(t, x, y)]



# Condiciones iniciales
x0 = 2.0
y0 = 0.0
z0 = [x0, y0]

# Intervalo de integración
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# Resolver la ecuación diferencial usando solve_ivp (método matemático)
sol = solve_ivp(system, t_span, z0, t_eval=t_eval)

# Condiciones iniciales para el método de Euler
x0_euler = 1.0
y0_euler = 0.0

# Tamaño del paso para el método de Euler
h_euler = 0.01

# Intervalo de integración para el método de Euler
t_end_euler = 20

# Calcular el número de pasos para el método de Euler
n_euler = int(t_end_euler / h_euler)

# Resolver la ecuación diferencial usando el método de Euler corregido
t_euler, x_euler, y_euler = euler_method2(dxdt, dydt, x0_euler, y0_euler, h_euler, n_euler)

# Graficar los resultados
plt.figure()
plt.plot(sol.t, sol.y[0], 'r-', label='Método Matemático (x)')
plt.plot(sol.t, sol.y[1], 'r--', label='Método Matemático (y)')
plt.plot(t_euler, x_euler, 'b-', label='Método de Euler (x)')
plt.plot(t_euler, y_euler, 'b--', label='Método de Euler (y)')
plt.xlabel('Tiempo (t)')
plt.ylabel('Solución')
plt.legend()
plt.title('Comparación del Método Matemático con el Método de Euler')
plt.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
