
"""
Mini-projet 2 : Optimisation industrielle sous contraintes
----------------------------------------------------------
Objectif :
- Minimiser le coût total d’un produit industriel
- Respecter des contraintes de poids, résistance et budget
- Analyser la sensibilité de la solution optimale

Outils :
- NumPy
- SciPy (optimisation)
- Matplotlib (visualisation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# ======================================================
# Classe de modélisation du processus industriel
# ======================================================

class ProcessusIndustriel:
    def __init__(self):
        # Coûts des matières premières (€/kg)
        self.cout_matiere = {
            'acier': 2.5,
            'aluminium': 3.0,
            'cuivre': 7.5,
            'plastique': 1.2
        }

        # Propriétés physiques des matériaux
        self.proprietes = {
            'acier': {'resistance': 250, 'cout_traitement': 0.5},
            'aluminium': {'resistance': 150, 'cout_traitement': 0.8},
            'cuivre': {'resistance': 210, 'cout_traitement': 1.2},
            'plastique': {'resistance': 50, 'cout_traitement': 0.3}
        }

        # Contraintes globales
        self.poids_max = 10.0       # kg
        self.resistance_min = 1000  # MPa
        self.budget_max = 50.0      # €

    # --------------------------------------------------
    # Fonction de coût
    # --------------------------------------------------
    def cout_total(self, x):
        """
        Fonction objectif à minimiser
        x = [acier, aluminium, cuivre, plastique] en kg
        """
        cout_matiere = sum(
            x[i] * list(self.cout_matiere.values())[i]
            for i in range(4)
        )

        cout_traitement = sum(
            x[i] * list(self.proprietes.values())[i]['cout_traitement']
            for i in range(4)
        )

        # Coût non linéaire (main-d’œuvre / complexité)
        cout_main_oeuvre = 20 * np.sqrt(sum(x))

        return cout_matiere + cout_traitement + cout_main_oeuvre

    # --------------------------------------------------
    # Optimisation
    # --------------------------------------------------
    def optimiser(self):
        # Point initial
        x0 = np.array([2.0, 2.0, 1.0, 1.0])

        # Bornes (kg)
        bounds = [(0.1, 5.0), (0.1, 5.0),
                  (0.1, 3.0), (0.1, 3.0)]

        # Contraintes
        constraints = [
            # Poids total
            {'type': 'ineq', 'fun': lambda x: self.poids_max - sum(x)},
            # Résistance minimale
            {'type': 'ineq', 'fun': lambda x:
                sum(x[i] * list(self.proprietes.values())[i]['resistance']
                    for i in range(4)) - self.resistance_min},
            # Budget
            {'type': 'ineq', 'fun': lambda x: self.budget_max - self.cout_total(x)},
            # Acier ≤ 50 %
            {'type': 'ineq', 'fun': lambda x: 0.5 * sum(x) - x[0]}
        ]

        # Quantité minimale de chaque matériau
        for i in range(4):
            constraints.append(
                {'type': 'ineq', 'fun': lambda x, i=i: x[i] - 0.1}
            )

        # Optimisation SLSQP
        result = optimize.minimize(
            self.cout_total,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result

    # --------------------------------------------------
    # Analyse de sensibilité
    # --------------------------------------------------
    def analyse_sensibilite(self, x_opt):
        sensibilite = {}

        for i, mat in enumerate(self.cout_matiere.keys()):
            variations = []
            for delta in [-0.1, 0.1]:
                x_pert = x_opt.copy()
                x_pert[i] += delta

                if x_pert[i] >= 0.1:
                    diff = self.cout_total(x_pert) - self.cout_total(x_opt)
                    variations.append(diff)

            sensibilite[mat] = np.mean(np.abs(variations))

        return sensibilite


# ======================================================
# Exécution du projet
# ======================================================

processus = ProcessusIndustriel()
resultat = processus.optimiser()

print("\n=== RÉSULTATS DE L'OPTIMISATION ===")
print("Succès :", resultat.success)
print("Message :", resultat.message)

materiaux = list(processus.cout_matiere.keys())
x_opt = resultat.x

for i, mat in enumerate(materiaux):
    print(f"{mat:10s} : {x_opt[i]:.3f} kg")

cout_final = processus.cout_total(x_opt)
poids_total = sum(x_opt)
resistance_totale = sum(
    x_opt[i] * processus.proprietes[materiaux[i]]['resistance']
    for i in range(4)
)

print(f"\nPoids total      : {poids_total:.2f} kg")
print(f"Résistance totale: {resistance_totale:.0f} MPa")
print(f"Coût total       : {cout_final:.2f} €")

# ======================================================
# Analyse de sensibilité
# ======================================================

sensibilite = processus.analyse_sensibilite(x_opt)

print("\n=== ANALYSE DE SENSIBILITÉ (±0.1 kg) ===")
for mat, val in sensibilite.items():
    print(f"{mat:10s} : variation moyenne = {val:.3f} €")

# ======================================================
# Visualisations
# ======================================================

# 1. Composition optimale
plt.figure(figsize=(6, 4))
plt.bar(materiaux, x_opt, color='steelblue')
plt.ylabel("Quantité (kg)")
plt.title("Composition optimale du produit")
plt.grid(axis='y', alpha=0.3)
plt.show()

# 2. Répartition des coûts
couts_matiere = [
    x_opt[i] * processus.cout_matiere[materiaux[i]]
    for i in range(4)
]
cout_traitement = sum(
    x_opt[i] * processus.proprietes[materiaux[i]]['cout_traitement']
    for i in range(4)
)
cout_main_oeuvre = 20 * np.sqrt(sum(x_opt))

labels = materiaux + ['Traitement', 'Main d’œuvre']
values = couts_matiere + [cout_traitement, cout_main_oeuvre]

plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Répartition du coût total")
plt.show()

# 3. Sensibilité
plt.figure(figsize=(6, 4))
plt.bar(sensibilite.keys(), sensibilite.values(), color='orange')
plt.ylabel("Variation moyenne du coût (€)")
plt.title("Sensibilité du coût aux matériaux")
plt.grid(axis='y', alpha=0.3)
plt.show()
