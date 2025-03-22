import numpy as np

def euler_method(f, x0, y0, h, n):
    """
    Aplica el método de Euler para resolver una ecuación diferencial ordinaria de primer orden.

    Args:
        f (callable): La función que define la derivada dy/dx, f(x, y).
        x0 (float): El valor inicial de x.
        y0 (float): El valor inicial de y.
        h (float): El tamaño del paso.
        n (int): El número de pasos.

    Returns:
        tuple: Una tupla que contiene dos arrays numpy, x e y, representando la solución numérica.

    Examples:
        >>> def f(x, y):
        ...     return y
        >>> x, y = euler_method(f, 0, 1, 0.1, 10)
        >>> np.allclose(y[-1], 2.5937424601)
        True
    """
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for i in range(1, n + 1):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
        x[i] = x[i - 1] + h
    return x, y

# Método de Euler corregido
def euler_method2(dxdt, dydt, x0, y0, h, n):
    """
    Implementación del método de Euler para resolver el sistema de ecuaciones diferenciales.

    Args:
        dxdt (function): Función que calcula dx/dt.
        dydt (function): Función que calcula dy/dt.
        x0 (float): Condición inicial para x.
        y0 (float): Condición inicial para y.
        h (float): Tamaño del paso.
        n (int): Número de pasos.

    Returns:
        t (ndarray): Arreglo de tiempos.
        x (ndarray): Arreglo de soluciones para x.
        y (ndarray): Arreglo de soluciones para y.
    """
    t = np.linspace(0, n * h, n + 1)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = x0
    y[0] = y0

    for i in range(1, n + 1):
        # Asegúrate de pasar tanto x como y a dxdt y dydt
        x[i] = x[i - 1] + h * dxdt(t[i - 1], x[i - 1], y[i - 1])
        y[i] = y[i - 1] + h * dydt(t[i - 1], x[i - 1], y[i - 1])

    return t, x, y

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # Pruebas unitarias adicionales (opcional)
    def test_euler_method():
        def f(x, y):
            return y
        x, y = euler_method(f, 0, 1, 0.1, 10)
        assert np.allclose(y[-1], 2.5937424601)

    def test_euler_method2():
        def dxdt(x, y):
            return y
        def dydt(x, y):
            return -x
        t, x, y = euler_method2(dxdt, dydt, 1, 0, 0.1, 10)
        assert np.allclose(x[-1], 0.5403020586813975)
        assert np.allclose(y[-1], -0.8414709848078965)

    test_euler_method()
    test_euler_method2()
    print("Todas las pruebas pasaron.")