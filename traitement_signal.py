# Traitement de signal

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Exemple 1: Génération de signaux
t = np.linspace(0, 10, 1000)

# Signal propre
signal_clean = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)

# Ajout de bruit
noise = np.random.normal(0, 0.5, len(t))
signal_noisy = signal_clean + noise

# Exemple 2: Filtrage
# Filtre passe-bas
sos_low = signal.butter(4, 3, 'low', fs=100, output='sos')
signal_filtered = signal.sosfilt(sos_low, signal_noisy)

# Filtre passe-bande
sos_band = signal.butter(4, [2, 6], 'band', fs=100, output='sos')
signal_band = signal.sosfilt(sos_band, signal_noisy)

# Exemple 3: Analyse spectrale
frequencies, psd = signal.welch(signal_noisy, fs=100, nperseg=256)

# Exemple 4: Détection de pics
peaks, properties = signal.find_peaks(signal_clean, 
                                      height=0.5, 
                                      distance=50)

# Exemple 5: Convolution
kernel = signal.gaussian(50, std=10)
signal_smoothed = signal.convolve(signal_noisy, kernel, mode='same')

# Visualisation
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Signaux originaux
axes[0, 0].plot(t, signal_clean, 'b-', label='Signal propre')
axes[0, 0].plot(t, signal_noisy, 'r-', alpha=0.5, label='Signal bruité')
axes[0, 0].set_title('Signaux originaux')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Signaux filtrés
axes[0, 1].plot(t, signal_filtered, 'g-', label='Filtre passe-bas (3 Hz)')
axes[0, 1].plot(t, signal_band, 'm-', label='Filtre passe-bande (2-6 Hz)')
axes[0, 1].set_title('Signaux filtrés')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Détection de pics
axes[1, 0].plot(t, signal_clean, 'b-')
axes[1, 0].plot(t[peaks], signal_clean[peaks], 'rx', markersize=10)
axes[1, 0].set_title('Détection de pics')
axes[1, 0].grid(True)

# Densité spectrale de puissance
axes[1, 1].semilogy(frequencies, psd)
axes[1, 1].set_xlabel('Fréquence (Hz)')
axes[1, 1].set_ylabel('PSD')
axes[1, 1].set_title('Densité Spectrale de Puissance')
axes[1, 1].grid(True)

# Convolution
axes[2, 0].plot(t, signal_noisy, 'r-', alpha=0.3, label='Signal bruité')
axes[2, 0].plot(t, signal_smoothed, 'b-', linewidth=2, label='Signal lissé')
axes[2, 0].set_title('Lissage par convolution')
axes[2, 0].legend()
axes[2, 0].grid(True)

# Réponse impulsionnelle des filtres
t_impulse = np.linspace(0, 1, 200)
impulse = np.zeros(200)
impulse[10] = 1

# Réponse des filtres
response_low = signal.sosfilt(sos_low, impulse)
response_band = signal.sosfilt(sos_band, impulse)

axes[2, 1].plot(t_impulse, response_low, 'g-', label='Passe-bas')
axes[2, 1].plot(t_impulse, response_band, 'm-', label='Passe-bande')
axes[2, 1].set_title('Réponses impulsionnelles')
axes[2, 1].legend()
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()