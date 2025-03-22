import numpy as np
import pytest
import time

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
    """
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
        x[i + 1] = x[i] + h
    return x, y

def f(x, y):
    return y

@pytest.mark.parametrize("n", [10, 100, 1000, 10000]) #parametrizacion de n.
def test_euler_method_benchmark(benchmark, n):
    benchmark.pedantic(euler_method, args=(f, 0, 1, 0.1, n), rounds=10, iterations=10) #benchmark pedantic

if __name__ == "__main__":
    pytest.main()