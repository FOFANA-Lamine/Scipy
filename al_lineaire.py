
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt

# Exemple 1: Décomposition de matrices
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

# Décomposition LU
P, L, U = linalg.lu(A)
print("Décomposition LU:")
print(f"P (matrice de permutation):\n{P}")
print(f"L (matrice triangulaire inférieure):\n{L}")
print(f"U (matrice triangulaire supérieure):\n{U}")

# Vérification
print(f"\nVérification (P@L@U == A): {(P@L@U).round(3) == A}")

# Décomposition QR
Q, R = linalg.qr(A)
print(f"\nQ (orthogonale):\n{Q.round(3)}")
print(f"R (triangulaire supérieure):\n{R.round(3)}")

# Décomposition en valeurs propres
eigenvalues, eigenvectors = linalg.eig(A)
print(f"\nValeurs propres: {eigenvalues}")
print(f"Vecteurs propres:\n{eigenvectors}")

# Exemple 2: Systèmes d'équations linéaires
b = np.array([1, 2, 3])

# Solution par différentes méthodes
x_lu = linalg.solve(A, b)
x_qr = linalg.solve_triangular(R, Q.T @ b)
x_cholesky = linalg.cho_solve(linalg.cho_factor(A), b)

print(f"\nSolutions du système Ax = b:")
print(f"LU: {x_lu}")
print(f"QR: {x_qr}")
print(f"Cholesky: {x_cholesky}")

# Exemple 3: Normes et conditionnement
norm_1 = linalg.norm(A, 1)
norm_2 = linalg.norm(A, 2)
norm_fro = linalg.norm(A, 'fro')
cond_number = np.linalg.cond(A)

print(f"\nNormes de la matrice A:")
print(f"Norme 1: {norm_1:.4f}")
print(f"Norme 2: {norm_2:.4f}")
print(f"Norme de Frobenius: {norm_fro:.4f}")
print(f"Conditionnement: {cond_number:.4f}")

# Visualisation
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Matrice originale
im0 = axes[0, 0].imshow(A, cmap='viridis')
axes[0, 0].set_title('Matrice A')
plt.colorbar(im0, ax=axes[0, 0])

# Matrices L et U
im1 = axes[0, 1].imshow(L, cmap='viridis')
axes[0, 1].set_title('Matrice L (LU)')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(U, cmap='viridis')
axes[0, 2].set_title('Matrice U (LU)')
plt.colorbar(im2, ax=axes[0, 2])

# Matrices Q et R
im3 = axes[1, 0].imshow(Q, cmap='viridis')
axes[1, 0].set_title('Matrice Q (QR)')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(R, cmap='viridis')
axes[1, 1].set_title('Matrice R (QR)')
plt.colorbar(im4, ax=axes[1, 1])

# Vecteurs propres
im5 = axes[1, 2].imshow(eigenvectors.real, cmap='RdBu')
axes[1, 2].set_title('Vecteurs propres')
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.show()