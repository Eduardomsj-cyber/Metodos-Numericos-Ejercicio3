import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definir la función f(x) = e^(-x) - x
def f(x):
    return np.exp(-x) - x

# Función de interpolación de Lagrange
def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if j != i:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Puntos de interpolación
x_points = np.array([0, 0.25, 0.5, 1])
y_points = f(x_points)

# Evaluar la interpolación de Lagrange en un rango de valores para graficar
x_vals = np.linspace(0, 1, 500)
y_vals = np.array([lagrange_interpolation(x_points, y_points, x) for x in x_vals])

# Graficar la función original y la interpolación de Lagrange
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = e^(-x) - x", color='blue')
plt.plot(x_vals, y_vals, label="Interpolación de Lagrange", linestyle='--', color='red')
plt.scatter(x_points, y_points, color='black', zorder=5, label="Puntos de Interpolación")
plt.legend()
plt.title("Interpolación de Lagrange para f(x) = e^(-x) - x")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# Función para encontrar la raíz usando el método de bisección
def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    iter_count = 0
    root_history = []  # Guardar las iteraciones para mostrar en la tabla
    error_history = []  # Guardar los errores para las gráficas
    while iter_count < max_iter:
        c = (a + b) / 2
        error = abs(f(c))  # Error en cada iteración
        root_history.append(c)
        error_history.append(error)
        if abs(f(c)) < tol:
            return c, iter_count, root_history, error_history
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
        iter_count += 1
    return (a + b) / 2, iter_count, root_history, error_history

# Función para encontrar la raíz de la interpolación de Lagrange
def find_root_lagrange(x_points, y_points, tol=1e-6, max_iter=100):
    # Usamos el método de bisección para encontrar la raíz de la interpolación de Lagrange
    root, iterations, root_history, error_history = bisection_method(
        lambda x: lagrange_interpolation(x_points, y_points, x), 0, 1, tol, max_iter)
    return root, iterations, root_history, error_history

# Encontrar la raíz de la interpolación
root, iterations, root_history, error_history = find_root_lagrange(x_points, y_points)
print(f"Raíz encontrada: {root}")
print(f"Iteraciones: {iterations}")

# Tabla con los valores obtenidos en cada iteración
iteration_table = pd.DataFrame({
    'Iteración': np.arange(1, len(root_history) + 1),
    'Raíz Aproximada': root_history,
    'Error Absoluto': error_history
})
print("\nTabla de iteraciones:")
print(iteration_table)

# Evaluar la convergencia del método de bisección y analizar los errores
errors = [abs(root_approx - root) for root_approx in root_history]

# Graficar los resultados de convergencia
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(root_history) + 1), errors, marker='o', color='red')
plt.yscale('log')
plt.xlabel("Iteración")
plt.ylabel("Error Absoluto")
plt.title("Convergencia de la raíz (Error Absoluto) en cada iteración")
plt.grid(True)
plt.show()

# Graficar la solución con la raíz aproximada
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = e^(-x) - x", color='blue')
plt.axvline(root, color='green', linestyle='--', label=f"Raíz Aproximada: {root:.6f}")
plt.legend()
plt.title("Solución aproximada de la raíz")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# Graficar los errores de la convergencia
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(error_history) + 1), error_history, marker='o', color='orange')
plt.yscale('log')
plt.xlabel("Iteración")
plt.ylabel("Error Absoluto")
plt.title("Evolución del error a lo largo de las iteraciones")
plt.grid(True)
plt.show()
