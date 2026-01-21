
# ============================================================
# Mini-projet 1 : Modélisation d’une épidémie (Modèle SIR)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ------------------------------------------------------------
# 1. Définition du modèle SIR
# ------------------------------------------------------------
def sir_model(y, t, beta, gamma):
    """
    Modèle SIR sous forme d'équations différentielles.
    
    y     : vecteur d'état [S, I, R]
    t     : temps
    beta  : taux de transmission
    gamma : taux de guérison
    """
    S, I, R = y
    N = S + I + R

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]


# ------------------------------------------------------------
# 2. Paramètres du modèle
# ------------------------------------------------------------
N = 10000        # population totale

I0 = 10          # infectés initiaux
R0 = 0           # rétablis initiaux
S0 = N - I0 - R0 # susceptibles initiaux

beta = 0.3       # taux de transmission
gamma = 0.1      # taux de guérison

y0 = [S0, I0, R0]

print("=== PARAMÈTRES DU MODÈLE ===")
print(f"Population totale N = {N}")
print(f"Taux de transmission β = {beta}")
print(f"Taux de guérison γ = {gamma}")
print(f"Nombre de reproduction R0 = {beta/gamma:.2f}")
print()


# ------------------------------------------------------------
# 3. Résolution numérique
# ------------------------------------------------------------
t = np.linspace(0, 160, 160)  # temps en jours

solution = odeint(sir_model, y0, t, args=(beta, gamma))
S, I, R = solution.T


# ------------------------------------------------------------
# 4. Analyse numérique de l’épidémie
# ------------------------------------------------------------
peak_I = np.max(I)
peak_day = t[np.argmax(I)]
final_infected = R[-1]

print("=== ANALYSE NUMÉRIQUE ===")
print(f"Pic d'infectés        : {peak_I:.0f} personnes")
print(f"Jour du pic           : {peak_day:.1f} jours")
print(f"Population infectée finale : {final_infected:.0f} personnes")
print()


# ------------------------------------------------------------
# 5. Visualisation principale S / I / R
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptibles', lw=2)
plt.plot(t, I, label='Infectés', lw=2)
plt.plot(t, R, label='Rétablis', lw=2)

plt.axvline(peak_day, color='red', linestyle='--',
            alpha=0.6, label='Pic épidémique')

plt.xlabel("Temps (jours)")
plt.ylabel("Nombre d'individus")
plt.title("Modèle SIR — Évolution d’une épidémie")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 6. Visualisation alternative (couleurs explicites)
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptibles', color='blue')
plt.plot(t, I, label='Infectés', color='red')
plt.plot(t, R, label='Rétablis', color='green')

plt.xlabel("Temps (jours)")
plt.ylabel("Nombre d'individus")
plt.title("Modèle SIR — Propagation d'une épidémie")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 7. Proportion de population infectée
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(t, I / N, label="Infectés (proportion)")
plt.xlabel("Temps (jours)")
plt.ylabel("Proportion de la population")
plt.title("Proportion d'individus infectés")
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. Interprétation finale (console)
# ------------------------------------------------------------
print("=== INTERPRÉTATION ===")
print("• L'épidémie démarre avec une croissance rapide des infectés.")
print("• Le pic apparaît lorsque les susceptibles deviennent limitants.")
print("• À long terme, I → 0 et la population devient majoritairement rétablie.")
print("• R0 > 1 indique une épidémie active.")
