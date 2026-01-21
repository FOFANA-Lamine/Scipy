
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt

# Exemple 1: Calcul de distances
points = np.array([[0, 0],
                   [1, 1],
                   [2, 2],
                   [3, 3],
                   [4, 4]])

# Matrice des distances
distance_matrix = spatial.distance_matrix(points, points)
print("Matrice des distances:")
print(distance_matrix)

# Distances spécifiques
euclidean_dist = spatial.distance.euclidean(points[0], points[4])
manhattan_dist = spatial.distance.cityblock(points[0], points[4])
cosine_dist = spatial.distance.cosine(points[0], points[4])

print(f"\nDistances entre le premier et dernier point:")
print(f"Euclidienne: {euclidean_dist:.4f}")
print(f"Manhattan: {manhattan_dist:.4f}")
print(f"Cosine: {cosine_dist:.4f}")

# Exemple 2: Diagramme de Voronoï
np.random.seed(42)
points_vor = np.random.rand(20, 2)

vor = spatial.Voronoi(points_vor)

# Exemple 3: Enveloppe convexe
hull = spatial.ConvexHull(points_vor)

# Exemple 4: Arbre KD pour recherches spatiales
kdtree = spatial.KDTree(points_vor)

# Recherche des plus proches voisins
query_point = np.array([[0.5, 0.5]])
distances, indices = kdtree.query(query_point, k=3)

print(f"\n3 plus proches voisins de {query_point[0]}:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i+1}. Point {points_vor[idx]} (distance: {dist:.4f})")

# Exemple 5: Triangulation de Delaunay
delaunay = spatial.Delaunay(points_vor)

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Diagramme de Voronoï
spatial.voronoi_plot_2d(vor, ax=axes[0, 0])
axes[0, 0].plot(points_vor[:, 0], points_vor[:, 1], 'ko')
axes[0, 0].set_title('Diagramme de Voronoï')
axes[0, 0].set_aspect('equal')

# Enveloppe convexe
axes[0, 1].plot(points_vor[:, 0], points_vor[:, 1], 'ko')
for simplex in hull.simplices:
    axes[0, 1].plot(points_vor[simplex, 0], points_vor[simplex, 1], 'r-')
axes[0, 1].plot(points_vor[hull.vertices, 0], 
                points_vor[hull.vertices, 1], 'r--', lw=2)
axes[0, 1].plot(points_vor[hull.vertices[0], 0], 
                points_vor[hull.vertices[0], 1], 'ro')
axes[0, 1].set_title('Enveloppe Convexe')
axes[0, 1].set_aspect('equal')

# Triangulation de Delaunay
axes[1, 0].triplot(points_vor[:, 0], points_vor[:, 1], delaunay.simplices)
axes[1, 0].plot(points_vor[:, 0], points_vor[:, 1], 'o')
axes[1, 0].set_title('Triangulation de Delaunay')
axes[1, 0].set_aspect('equal')

# Recherche des plus proches voisins
axes[1, 1].plot(points_vor[:, 0], points_vor[:, 1], 'ko', alpha=0.5)
axes[1, 1].plot(query_point[0, 0], query_point[0, 1], 'ro', 
                markersize=10, label='Point de requête')

# Affichage des plus proches voisins
for dist, idx in zip(distances[0], indices[0]):
    circle = plt.Circle(query_point[0], dist, color='r', 
                       fill=False, linestyle='--', alpha=0.5)
    axes[1, 1].add_patch(circle)
    axes[1, 1].plot(points_vor[idx, 0], points_vor[idx, 1], 'go', 
                   markersize=8, label=f'Voisin {idx}')

axes[1, 1].set_title('Recherche des plus proches voisins (KD-Tree)')
axes[1, 1].legend()
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.show()

# Mesures géométriques
print(f"\nMesures de l'enveloppe convexe:")
print(f"Volume: {hull.volume:.4f}")
print(f"Surface: {hull.area:.4f}")
print(f"Nombre de sommets: {len(hull.vertices)}")