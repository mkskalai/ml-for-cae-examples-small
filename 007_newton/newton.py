import numpy as np
import imageio
import matplotlib.pyplot as plt

# Define the non-convex function
def f(x):
    return x**3 - 2*x**2 + x + 1

# Derivative of the function
def df(x):
    return 3*x**2 - 4*x + 1

# Newtonian optimization to find root
def newton_optimization(func, derivative, x0, tol=1e-6, max_iter=100):
    x_values = [x0]
    for _ in range(max_iter):
        x_new = x_values[-1] - func(x_values[-1]) / derivative(x_values[-1])
        x_values.append(x_new)
        if abs(func(x_new)) < tol:
            break
    return x_values

# Generate GIF frames
def generate_gif():
    x_values = newton_optimization(f, df, x0=1.5)

    for i in range(1, len(x_values)):
        fig, ax = plt.subplots()
        x_range = np.linspace(-5.5, 5.0, 100)
        y_range = f(x_range)
        ax.plot(x_range, y_range, label='Function: $x^3 - 2x^2 + x + 1$')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', label='y=0')
        ax.plot(x_values[i - 1], f(x_values[i - 1]), 'ro')  # Previous point
        ax.plot(x_values[i], f(x_values[i]), 'go')          # Current point
        ax.plot([x_values[i - 1], x_values[i]], [f(x_values[i - 1]), 0], 'k-')  # Line to x-axis
        ax.text(x_values[i] + 0.05, f(x_values[i]), f'Iter {i}', fontsize=14)

        ax.set_title('Newtonian Optimization')
        ax.legend()
        ax.grid(True)

        # Save frame
        plt.savefig(f'newton_frame_{i}.png')
        plt.clf()
        
        
if __name__ == "__main__":
    generate_gif()