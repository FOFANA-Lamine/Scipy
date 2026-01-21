# Traitement d'image


from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import data  # Pour des images d'exemple  ( pip install scikit-image)


"""

from skimage import data
import matplotlib.pyplot as plt

# Charger une image d'exemple
image = data.camera()   # ou data.camera() ,data.astronaut(), data.coins(), etc.

# Afficher l'image
plt.imshow(image)
plt.axis('off')
plt.show()
"""

# Chargement d'une image d'exemple
image = data.camera()   # ou data.camera() ,data.astronaut(), data.coins(), etc.

# Exemple 1: Filtrage
# Filtre gaussien
image_gaussian = ndimage.gaussian_filter(image, sigma=2)

# Filtre médian
image_median = ndimage.median_filter(image, size=3)

# Exemple 2: Morphologie mathématique
structure = ndimage.generate_binary_structure(2, 2)
image_eroded = ndimage.binary_erosion(image > 128, structure=structure)
image_dilated = ndimage.binary_dilation(image > 128, structure=structure)

# Exemple 3: Mesures d'objets
labeled_image, num_features = ndimage.label(image > 128)
sizes = ndimage.sum(image > 128, labeled_image, range(1, num_features + 1))

# Exemple 4: Rotation et transformation
image_rotated = ndimage.rotate(image, angle=45, reshape=False)
image_shifted = ndimage.shift(image, shift=(20, 15))

# Exemple 5: Détection de contours
sobel_x = ndimage.sobel(image, axis=0)
sobel_y = ndimage.sobel(image, axis=1)
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Visualisation
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Image originale et filtres
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Originale')

axes[0, 1].imshow(image_gaussian, cmap='gray')
axes[0, 1].set_title('Filtre Gaussien')

axes[0, 2].imshow(image_median, cmap='gray')
axes[0, 2].set_title('Filtre Médian')

# Morphologie
axes[1, 0].imshow(image_eroded, cmap='gray')
axes[1, 0].set_title('Érosion')

axes[1, 1].imshow(image_dilated, cmap='gray')
axes[1, 1].set_title('Dilatation')

# Rotation et translation
axes[1, 2].imshow(image_rotated, cmap='gray')
axes[1, 2].set_title('Rotation 45°')

axes[2, 0].imshow(image_shifted, cmap='gray')
axes[2, 0].set_title('Translation')

# Détection de contours
axes[2, 1].imshow(gradient_magnitude, cmap='hot')
axes[2, 1].set_title('Gradient (Sobel)')

# Objets labellisés
axes[2, 2].imshow(labeled_image, cmap='nipy_spectral')
axes[2, 2].set_title(f'Objets détectés: {num_features}')

plt.tight_layout()
plt.show()

# Analyse quantitative
print(f"Nombre d'objets détectés: {num_features}")
print(f"Taille des objets: {sizes[:10]}")  # Affiche les 10 premiers
print(f"Taille moyenne: {np.mean(sizes):.2f}")
print(f"Taille maximale: {np.max(sizes)}")