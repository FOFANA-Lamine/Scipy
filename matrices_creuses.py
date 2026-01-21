# Matrices creuses

from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

# Exemple 1: Création de matrices creuses
# Matrice dense pour comparaison
dense_matrix = np.array([[1, 0, 0, 2],
                         [0, 0, 3, 0],
                         [0, 4, 0, 0],
                         [5, 0, 0, 6]])

# Formats de matrices creuses
csr_matrix = sparse.csr_matrix(dense_matrix)  # Compressed Sparse Row
csc_matrix = sparse.csc_matrix(dense_matrix)  # Compressed Sparse Column
coo_matrix = sparse.coo_matrix(dense_matrix)  # Coordinate format

print("Matrice dense:")
print(dense_matrix)
print(f"\nTaille dense: {dense_matrix.nbytes} bytes")
print(f"Taille CSR: {csr_matrix.data.nbytes + csr_matrix.indices.nbytes + csr_matrix.indptr.nbytes} bytes")

# Exemple 2: Opérations sur matrices creuses
# Création d'une grande matrice creuse
n = 1000
sparse_large = sparse.random(n, n, density=0.01, format='csr')

# Multiplication matricielle
result_sparse = sparse_large.dot(sparse_large.T)

# Résolution de système linéaire
b = np.random.rand(n)
# x = sparse.linalg.spsolve(sparse_large, b) 
x = np.linalg.solve(sparse_large.toarray(), b)


print(f"\nMatrice {n}x{n} avec densité 1%")
print(f"Éléments non-nuls: {sparse_large.nnz}")
print(f"Taux de compression: {(1 - sparse_large.nnz/(n*n))*100:.2f}%")

# Exemple 3: Construction efficace
# Construction COO
rows = np.array([0, 0, 1, 2, 3, 3])
cols = np.array([0, 3, 2, 1, 0, 3])
data = np.array([1, 2, 3, 4, 5, 6])

coo_const = sparse.coo_matrix((data, (rows, cols)), shape=(4, 4))

# Conversion entre formats
csr_from_coo = coo_const.tocsr()
csc_from_csr = csr_from_coo.tocsc()

# Visualisation
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Matrices dans différents formats
formats = ['Dense', 'CSR', 'CSC', 'COO']
matrices = [dense_matrix, csr_matrix.toarray(), 
            csc_matrix.toarray(), coo_matrix.toarray()]

for i, (title, mat) in enumerate(zip(formats, matrices)):
    ax = axes[i//3, i%3]
    im = ax.imshow(mat, cmap='viridis')
    ax.set_title(f'Format {title}')
    plt.colorbar(im, ax=ax)

# Motif de la grande matrice creuse
axes[1, 2].spy(sparse_large, markersize=1)
axes[1, 2].set_title(f'Matrice creuse {n}x{n} (1% remplissage)')

plt.tight_layout()
plt.show()

# Comparaison des performances
import time

# Création de matrices de grande taille
n_large = 2000
dense_large = np.random.rand(n_large, n_large)
sparse_large = sparse.random(n_large, n_large, density=0.01, format='csr')

# Multiplication - Dense
start = time.time()
result_dense = dense_large.dot(dense_large)
time_dense = time.time() - start

# Multiplication - Sparse
start = time.time()
result_sparse = sparse_large.dot(sparse_large)
time_sparse = time.time() - start

print(f"\nComparaison des performances (n={n_large}):")
print(f"Temps multiplication dense: {time_dense:.4f} s")
print(f"Temps multiplication creuse: {time_sparse:.4f} s")
print(f"Accélération: {time_dense/time_sparse:.2f}x")