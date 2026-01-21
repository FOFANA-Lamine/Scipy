
from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt

# Exemple 1: Minimisation simple
def rosenbrock(x):
    """Fonction de Rosenbrock (test classique en optimisation)"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Point de départ
x0 = np.array([-1, 1])

# Minimisation avec différentes méthodes
methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS']
results = {}

for method in methods:
    result = optimize.minimize(rosenbrock, x0, method=method)
    results[method] = result.x
    print(f"{method}: x = {result.x}, f(x) = {result.fun:.2e}")

# Exemple 2: Moindres carrés non-linéaires
def model(params, x):
    a, b, c = params
    return a * np.exp(-b * x) + c

x_data = np.linspace(0, 4, 50)
y_data = 2.5 * np.exp(-1.3 * x_data) + 0.5 + 0.2 * np.random.randn(50)

def residuals(params, x, y):
    return y - model(params, x)

initial_guess = [1, 1, 1]
result_lsq = optimize.least_squares(residuals, initial_guess, args=(x_data, y_data))
print(f"\nParamètres optimaux: {result_lsq.x}")

# Exemple 3: Optimisation avec contraintes
def objective(x):
    return x[0]**2 + x[1]**2

# Contraintes: x[0] + x[1] >= 1, x[0] >= 0, x[1] >= 0
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1},
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1]}
]

result_const = optimize.minimize(objective, [0.5, 0.5], 
                                 constraints=constraints)
print(f"\nAvec contraintes: x = {result_const.x}, f(x) = {result_const.fun}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Surface de Rosenbrock
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock([X, Y])

ax1 = axes[0]
contour = ax1.contour(X, Y, Z, levels=50, cmap='viridis')
ax1.scatter(1, 1, c='r', s=100, marker='*', label='Minimum global')
for method, point in results.items():
    ax1.scatter(point[0], point[1], label=method, s=50)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Fonction de Rosenbrock')
ax1.legend()

# Ajustement par moindres carrés
ax2 = axes[1]
ax2.scatter(x_data, y_data, alpha=0.5, label='Données')
x_fit = np.linspace(0, 4, 100)
y_fit = model(result_lsq.x, x_fit)
ax2.plot(x_fit, y_fit, 'r-', linewidth=2, label='Ajustement')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Ajustement non-linéaire')
ax2.legend()

plt.tight_layout()
plt.show()