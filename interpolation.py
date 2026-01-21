
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

# Données d'exemple
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 2, 1, 3, 2, 4])

# Différentes méthodes d'interpolation
x_new = np.linspace(0, 5, 100)

# Interpolation linéaire
f_linear = interpolate.interp1d(x, y, kind='linear')

# Interpolation cubique
f_cubic = interpolate.interp1d(x, y, kind='cubic')

# Splines lissées
spline = interpolate.UnivariateSpline(x, y, s=0.5)

# Interpolation 2D (exemple)
def f_2d(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

x_2d = np.linspace(-5, 5, 20)
y_2d = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_2d, y_2d)
Z = f_2d(X, Y)

# Interpolation 2D
interp_2d = interpolate.RectBivariateSpline(x_2d, y_2d, Z)

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1D Interpolation
axes[0, 0].scatter(x, y, s=50, label='Points originaux')
axes[0, 0].plot(x_new, f_linear(x_new), 'r-', label='Linéaire')
axes[0, 0].plot(x_new, f_cubic(x_new), 'g--', label='Cubique')
axes[0, 0].plot(x_new, spline(x_new), 'b:', label='Spline')
axes[0, 0].set_title('Interpolation 1D')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Surface originale
axes[0, 1].contourf(X, Y, Z, 20, cmap='viridis')
axes[0, 1].set_title('Surface originale')

# Surface interpolée
x_new_2d = np.linspace(-5, 5, 100)
y_new_2d = np.linspace(-5, 5, 100)
Z_interp = interp_2d(x_new_2d, y_new_2d)
axes[1, 0].contourf(x_new_2d, y_new_2d, Z_interp, 20, cmap='viridis')
axes[1, 0].set_title('Surface interpolée')

# Différence
axes[1, 1].contourf(x_new_2d, y_new_2d, 
                    Z_interp - f_2d(*np.meshgrid(x_new_2d, y_new_2d)), 
                    20, cmap='RdBu')
axes[1, 1].set_title('Erreur d interpolation')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')

plt.colorbar(axes[1, 1].contourf(x_new_2d, y_new_2d, 
                                 Z_interp - f_2d(*np.meshgrid(x_new_2d, y_new_2d)), 
                                 20, cmap='RdBu'), ax=axes[1, 1])
plt.tight_layout()
plt.show()