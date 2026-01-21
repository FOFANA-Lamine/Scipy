# Module SciPy : scipy.integrate - Intégration Numérique

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
# Installation avec les dépendances recommandées
# pip install numpy matplotlib pandas scipy
print(f"Version de SciPy : {scipy.__version__}")


# Exemple 1: Intégration simple
def f(x):
    return np.sin(x) + np.cos(x)

# Intégration quadratique
result, error = integrate.quad(f, 0, np.pi)
print(f"∫sin(x)+cos(x) dx de 0 à π = {result:.4f}")
print(f"Erreur estimée: {error:.2e}")

# Exemple 2: Intégration double
def f2(x, y):
    return np.exp(-(x**2 + y**2))

# ∫∫ e^(-x²-y²) dx dy
result_dbl = integrate.dblquad(f2, -1, 1, lambda x: -1, lambda x: 1)
print(f"\nIntégrale double: {result_dbl[0]:.4f}")

# Exemple 3: Équations différentielles ordinaires (EDO)
# Équation: dy/dt = -2y, avec y(0)=1
def dydt(y, t):
    return -2 * y

t = np.linspace(0, 5, 100)
y0 = 1  # Condition initiale

solution = integrate.odeint(dydt, y0, t)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
x = np.linspace(0, np.pi, 100)
plt.plot(x, f(x), 'b-', label='f(x)=sin(x)+cos(x)')
plt.fill_between(x, f(x), alpha=0.3)
plt.title("Fonction à intégrer")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, solution, 'r-', linewidth=2, label='Solution EDO')
plt.plot(t, np.exp(-2*t), 'k--', label='Solution analytique')
plt.title("Équation différentielle")
plt.legend()
plt.tight_layout()
plt.show()


