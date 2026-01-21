


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# -------------------------------
# Paramètres du signal
# -------------------------------
fs = 1000            # Fréquence d'échantillonnage (Hz)
T = 1.0              # Durée du signal (seconde)
N = fs               # Nombre d'échantillons

t = np.linspace(0, T, N, endpoint=False)

# -------------------------------
# Signal temporel
# -------------------------------
# Signal composé de deux sinusoïdes
signal_time = (
    np.sin(2 * np.pi * 50 * t) +
    0.5 * np.sin(2 * np.pi * 120 * t)
)

# -------------------------------
# Transformée de Fourier
# -------------------------------
signal_fft = fft(signal_time)

# Axe des fréquences
freqs = fftfreq(N, 1/fs)

# Amplitude normalisée du spectre
amplitude = np.abs(signal_fft) / N

# Partie positive du spectre
positive_freqs = freqs[:N // 2]
positive_amplitude = amplitude[:N // 2]

# -------------------------------
# Fréquences dominantes
# -------------------------------
indices = np.argsort(positive_amplitude)[-5:]
dominant_freqs = positive_freqs[indices]

print("Fréquences dominantes détectées (Hz) :")
for f in sorted(dominant_freqs):
    print(f"  - {f:.1f} Hz")

# -------------------------------
# Visualisation
# -------------------------------
plt.figure(figsize=(12, 5))

# Signal temporel
plt.subplot(1, 2, 1)
plt.plot(t, signal_time)
plt.title("Signal temporel")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Spectre fréquentiel
plt.subplot(1, 2, 2)
plt.stem(positive_freqs, positive_amplitude, use_line_collection=True)
plt.title("Spectre fréquentiel (FFT)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 200)
plt.grid(True)

plt.tight_layout()
plt.show()
