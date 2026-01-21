

"""
Optimisation d'un processus industriel :
- Minimisation des coûts
- Contraintes de production
- Analyse de sensibilité
"""
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt  

class ProcessusIndustriel:
    def __init__(self):
        # Coûts des matières premières (€/kg)
        self.cout_matiere = {
            'acier': 2.5,
            'aluminium': 3.0,
            'cuivre': 7.5,
            'plastique': 1.2
        }
        
        # Propriétés des matériaux
        self.proprietes = {
            'acier': {'densite': 7.85, 'resistance': 250, 'cout_traitement': 0.5},
            'aluminium': {'densite': 2.7, 'resistance': 150, 'cout_traitement': 0.8},
            'cuivre': {'densite': 8.96, 'resistance': 210, 'cout_traitement': 1.2},
            'plastique': {'densite': 1.2, 'resistance': 50, 'cout_traitement': 0.3}
        }
        
        # Contraintes
        self.poids_max = 10.0  # kg
        self.resistance_min = 1000  # MPa
        self.budget_max = 50.0  # €
        self.production_min = 100  # unités/jour
        
    def cout_total(self, x):
        """Fonction de coût à minimiser"""
        # x = [acier, aluminium, cuivre, plastique] en kg
        cout_matiere = sum(x[i] * list(self.cout_matiere.values())[i] 
                          for i in range(4))
        cout_traitement = sum(x[i] * list(self.proprietes.values())[i]['cout_traitement'] 
                            for i in range(4))
        cout_main_oeuvre = 20 * np.sqrt(sum(x))  # Coût non-linéaire
        return cout_matiere + cout_traitement + cout_main_oeuvre
    
    def contraintes(self, x):
        """Fonctions de contraintes"""
        contraintes = []
        
        # 1. Poids total
        poids_total = sum(x)
        contraintes.append(self.poids_max - poids_total)  # ≤ 0
        
        # 2. Résistance minimale
        resistance = sum(x[i] * list(self.proprietes.values())[i]['resistance'] 
                        for i in range(4))
        contraintes.append(resistance - self.resistance_min)  # ≥ 0
        
        # 3. Budget
        cout = self.cout_total(x)
        contraintes.append(self.budget_max - cout)  # ≤ 0
        
        # 4. Proportions minimales (chaque matériau ≥ 0.1 kg)
        for i in range(4):
            contraintes.append(x[i] - 0.1)  # ≥ 0
        
        # 5. Équilibre des matériaux (acier ≤ 50% du total)
        contraintes.append(0.5 * sum(x) - x[0])  # ≥ 0
        
        return contraintes
    
    def optimiser(self):
        """Optimisation du processus"""
        # Point de départ
        x0 = np.array([2.0, 2.0, 1.0, 1.0])
        
        # Bornes (kg)
        bounds = [(0.1, 5.0), (0.1, 5.0), (0.1, 3.0), (0.1, 3.0)]
        
        # Définition des contraintes
        constraints = [
            {'type': 'ineq', 'fun': lambda x: self.poids_max - sum(x)},
            {'type': 'ineq', 'fun': lambda x: 
             sum(x[i] * list(self.proprietes.values())[i]['resistance'] 
                 for i in range(4)) - self.resistance_min},
            {'type': 'ineq', 'fun': lambda x: self.budget_max - self.cout_total(x)},
            {'type': 'ineq', 'fun': lambda x: 0.5 * sum(x) - x[0]}
        ]
        
        # Ajout des contraintes de bornes inférieures
        for i in range(4):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - 0.1})
        
        # Optimisation
        result = optimize.minimize(
            self.cout_total,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        return result
    
    def analyse_sensibilite(self, x_opt):
        """Analyse de sensibilité autour de la solution optimale"""
        perturbations = []
        
        # Perturbation de chaque variable
        for i in range(4):
            for delta in [-0.1, 0.1]:
                x_pert = x_opt.copy()
                x_pert[i] += delta
                if x_pert[i] >= 0.1:  # Respect des bornes
                    cout_pert = self.cout_total(x_pert)
                    perturbations.append({
                        'variable': list(self.cout_matiere.keys())[i],
                        'delta': delta,
                        'cout': cout_pert,
                        'variation': cout_pert - self.cout_total(x_opt)
                    })
        
        return perturbations
    
    def visualiser(self, result, perturbations):
        """Visualisation des résultats"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Solution optimale
        materiaux = list(self.cout_matiere.keys())
        x_opt = result.x
        
        # Diagramme en barres des quantités optimales
        ax1 = axes[0, 0]
        bars = ax1.bar(materiaux, x_opt, color=['steelblue', 'silver', 'brown', 'lightgreen'])
        ax1.set_ylabel('Quantité (kg)')
        ax1.set_title('Composition optimale')
        ax1.set_ylim(0, max(x_opt) * 1.2)
        
        # Ajout des valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Répartition des coûts
        couts = [
            x_opt[i] * self.cout_matiere[materiaux[i]] 
            for i in range(4)
        ]
        cout_traitement = sum(
            x_opt[i] * self.proprietes[materiaux[i]]['cout_traitement'] 
            for i in range(4)
        )
        cout_main_oeuvre = 20 * np.sqrt(sum(x_opt))
        
        cout_labels = materiaux + ['Traitement', 'Main d\'œuvre']
        cout_values = couts + [cout_traitement, cout_main_oeuvre]
        
        ax2 = axes[0, 1]
        colors2 = plt.cm.Set3(np.linspace(0, 1, len(cout_labels)))
        wedges, texts, autotexts = ax2.pie(cout_values, labels=cout_labels, 
                                          autopct='%1.1f%%', colors=colors2,
                                          startangle=90)
        ax2.set_title('Répartition des coûts')
        
        # Propriétés du produit final
        ax3 = axes[0, 2]
        proprietes_finales = {
            'Poids (kg)': sum(x_opt),
            'Résistance (MPa)': sum(x_opt[i] * self.proprietes[materiaux[i]]['resistance'] 
                                   for i in range(4)),
            'Coût total (€)': self.cout_total(x_opt),
            'Coût unitaire (€/kg)': self.cout_total(x_opt) / sum(x_opt)
        }
        
        y_pos = range(len(proprietes_finales))
        ax3.barh(y_pos, list(proprietes_finales.values()))
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(list(proprietes_finales.keys()))
        ax3.set_xlabel('Valeur')
        ax3.set_title('Propriétés du produit final')
        
        # Analyse de sensibilité
        ax4 = axes[1, 0]
        sens_data = {}
        for p in perturbations:
            if p['variable'] not in sens_data:
                sens_data[p['variable']] = []
            sens_data[p['variable']].append(abs(p['variation']))
        
        variables_sens = list(sens_data.keys())
        sensibilites = [np.mean(vals) for vals in sens_data.values()]
        
        bars_sens = ax4.bar(variables_sens, sensibilites, 
                           color=['red' if s > 1 else 'blue' for s in sensibilites])
        ax4.set_ylabel('Variation moyenne du coût (€)')
        ax4.set_title('Sensibilité aux variations (±0.1 kg)')
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # Convergence de l'optimisation
        ax5 = axes[1, 1]
        if hasattr(result, 'nit'):
            iterations = list(range(1, result.nit + 1))
            # Simulation de l'historique des coûts (simplifiée)
            couts_hist = [self.cout_total(x_opt * (1 - i/result.nit) + x_opt * (i/result.nit)) 
                         for i in iterations]
            ax5.plot(iterations, couts_hist, 'b-o', linewidth=2)
            ax5.set_xlabel('Itération')
            ax5.set_ylabel('Coût (€)')
            ax5.set_title('Convergence de l\'optimisation')
            ax5.grid(True, alpha=0.3)
        
        # Espace des solutions faisables (projection 2D)
        ax6 = axes[1, 2]
        
        # Génération de points aléatoires dans l'espace faisable
        n_points = 500
        points_faisables = []
        couts_faisables = []
        
        for _ in range(n_points):
            x_test = np.random.uniform([0.1, 0.1, 0.1, 0.1], 
                                       [5.0, 5.0, 3.0, 3.0])
            if all(c >= 0 for c in self.contraintes(x_test)[:3]):  # Contraintes principales
                points_faisables.append(x_test[:2])  # Acier et aluminium seulement
                couts_faisables.append(self.cout_total(x_test))
        
        if points_faisables:
            points_faisables = np.array(points_faisables)
            scatter = ax6.scatter(points_faisables[:, 0], points_faisables[:, 1],
                                 c=couts_faisables, cmap='viridis', 
                                 alpha=0.6, s=20)
            ax6.scatter(x_opt[0], x_opt[1], c='red', s=200, 
                       marker='*', label='Solution optimale')
            ax6.set_xlabel('Acier (kg)')
            ax6.set_ylabel('Aluminium (kg)')
            ax6.set_title('Espace des solutions faisables')
            ax6.legend()
            plt.colorbar(scatter, ax=ax6, label='Coût (€)')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Exécution du cas d'optimisation
processus = ProcessusIndustriel()
resultat = processus.optimiser()

print("=== RÉSULTATS DE L'OPTIMISATION ===")
print(f"Succès: {resultat.success}")
print(f"Message: {resultat.message}")
print(f"\nSolution optimale:")
materiaux = list(processus.cout_matiere.keys())
for i, mat in enumerate(materiaux):
    print(f"  {mat}: {resultat.x[i]:.3f} kg")

print(f"\nCoût total: {processus.cout_total(resultat.x):.2f} €")
print(f"Poids total: {sum(resultat.x):.2f} kg")
print(f"Résistance: {sum(resultat.x[i] * processus.proprietes[mat]['resistance'] for i, mat in enumerate(materiaux)):.0f} MPa")

print(f"\nVérification des contraintes:")
contraintes = processus.contraintes(resultat.x)
print(f"  Poids ≤ {processus.poids_max} kg: {contraintes[0] >= 0}")
print(f"  Résistance ≥ {processus.resistance_min} MPa: {contraintes[1] >= 0}")
print(f"  Budget ≤ {processus.budget_max} €: {contraintes[2] >= 0}")

# Analyse de sensibilité
print("\n=== ANALYSE DE SENSIBILITÉ ===")
perturbations = processus.analyse_sensibilite(resultat.x)
print("Impact d'une variation de ±0.1 kg sur le coût:")
for p in perturbations[:8]:  # Afficher les 8 premières perturbations
    print(f"  {p['variable']} {p['delta']:+}: {p['variation']:+.3f} €")

# Visualisation
fig = processus.visualiser(resultat, perturbations)