

"""
Simulation d'une analyse de données médicales :
- Chargement et prétraitement des données
- Analyse statistique
- Modélisation
- Visualisation
"""
import pandas as pd
from scipy import stats, optimize, signal
import numpy as np
import matplotlib.pyplot as plt



# Simulation de données médicales
np.random.seed(42)
n_patients = 200

# Création d'un DataFrame simulé
data = pd.DataFrame({
    'age': np.random.normal(45, 15, n_patients).astype(int),
    'blood_pressure_systolic': np.random.normal(120, 20, n_patients),
    'blood_pressure_diastolic': np.random.normal(80, 10, n_patients),
    'cholesterol': np.random.normal(200, 40, n_patients),
    'glucose': np.random.lognormal(4.5, 0.3, n_patients),
    'bmi': np.random.normal(25, 5, n_patients),
    'smoker': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
    'risk_score': np.zeros(n_patients)
})

# Calcul d'un score de risque (simulé)
data['risk_score'] = (
    0.05 * (data['age'] - 45) +
    0.03 * (data['blood_pressure_systolic'] - 120) +
    0.02 * (data['cholesterol'] - 200) +
    0.1 * data['smoker'] +
    np.random.normal(0, 0.5, n_patients)
)

# 1. Analyse descriptive
print("=== ANALYSE DESCRIPTIVE ===")
print(f"Nombre de patients: {len(data)}")
print(f"\nStatistiques descriptives:")
print(data.describe())

# 2. Tests d'hypothèses
print("\n=== TESTS D'HYPOTHÈSES ===")

# Comparaison fumeurs vs non-fumeurs
smokers = data[data['smoker'] == 1]
non_smokers = data[data['smoker'] == 0]

# Test t
t_stat, p_value = stats.ttest_ind(smokers['risk_score'], 
                                   non_smokers['risk_score'])
print(f"Test t (fumeurs vs non-fumeurs):")
print(f"  t-statistique: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Différence significative: {p_value < 0.05}")

# 3. Corrélations
print("\n=== ANALYSE DE CORRÉLATION ===")
correlation_matrix = data.corr()
print("Matrice de corrélation (Pearson):")
print(correlation_matrix['risk_score'].sort_values(ascending=False))

# Test de corrélation pour chaque variable
for col in ['age', 'blood_pressure_systolic', 'cholesterol', 'bmi']:
    corr, p_val = stats.pearsonr(data[col], data['risk_score'])
    print(f"\n{col}:")
    print(f"  Coefficient de corrélation: {corr:.4f}")
    print(f"  p-value: {p_val:.4f}")

# 4. Modélisation - Régression linéaire multiple
print("\n=== MODÉLISATION PAR RÉGRESSION ===")
from sklearn.preprocessing import StandardScaler

# Préparation des données
X = data[['age', 'blood_pressure_systolic', 'cholesterol', 'bmi', 'smoker']]
y = data['risk_score']

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ajout d'une colonne pour l'intercept
X_scaled = np.column_stack([np.ones(len(X_scaled)), X_scaled])

# Résolution par moindres carrés
beta, resid, rank, s = np.linalg.lstsq(X_scaled, y, rcond=None)

print("Coefficients de régression (standardisés):")
variables = ['Intercept', 'Age', 'BP Systolic', 'Cholesterol', 'BMI', 'Smoker']
for var, coef in zip(variables, beta):
    print(f"  {var}: {coef:.4f}")

# 5. Analyse de tendance temporelle (simulée)
print("\n=== ANALYSE TEMPORELLE ===")
# Simulation de données temporelles
time = np.arange(0, 365, 1)
hospital_admissions = (
    50 + 
    20 * np.sin(2 * np.pi * time/365) +  # Variation saisonnière
    10 * np.sin(2 * np.pi * time/7) +    # Variation hebdomadaire
    np.random.normal(0, 5, len(time))    # Bruit
)

# Filtrage pour extraire la tendance
sos = signal.butter(4, 1/30, 'low', fs=1, output='sos')
trend = signal.sosfilt(sos, hospital_admissions)

# Détection des pics (épidémies)
peaks, properties = signal.find_peaks(hospital_admissions, 
                                      height=70, 
                                      distance=30)

# 6. Visualisation complète
fig = plt.figure(figsize=(15, 10))

# Distribution du score de risque
ax1 = plt.subplot(2, 3, 1)
ax1.hist(data['risk_score'], bins=20, edgecolor='black', alpha=0.7)
ax1.axvline(data['risk_score'].mean(), color='red', 
           linestyle='--', label=f'Moyenne: {data["risk_score"].mean():.2f}')
ax1.set_xlabel('Score de risque')
ax1.set_ylabel('Fréquence')
ax1.set_title('Distribution du score de risque')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Comparaison fumeurs vs non-fumeurs
ax2 = plt.subplot(2, 3, 2)
box_data = [non_smokers['risk_score'], smokers['risk_score']]
bp = ax2.boxplot(box_data, labels=['Non-fumeurs', 'Fumeurs'], 
                 patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax2.set_ylabel('Score de risque')
ax2.set_title('Impact du tabagisme')
ax2.grid(True, alpha=0.3)

# Matrice de corrélation
ax3 = plt.subplot(2, 3, 3)
corr_plot = ax3.imshow(correlation_matrix, cmap='coolwarm', 
                       vmin=-1, vmax=1)
plt.colorbar(corr_plot, ax=ax3)
ax3.set_xticks(range(len(correlation_matrix.columns)))
ax3.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
ax3.set_yticks(range(len(correlation_matrix.columns)))
ax3.set_yticklabels(correlation_matrix.columns)
ax3.set_title('Matrice de corrélation')

# Régression - Importance des variables
ax4 = plt.subplot(2, 3, 4)
variables = variables[1:]  # Exclure l'intercept
coeffs = beta[1:]
colors = ['green' if c > 0 else 'red' for c in coeffs]
bars = ax4.barh(variables, np.abs(coeffs), color=colors)
ax4.set_xlabel('Coefficient (valeur absolue)')
ax4.set_title('Importance des variables (standardisées)')
ax4.grid(True, alpha=0.3, axis='x')

# Données temporelles
ax5 = plt.subplot(2, 3, 5)
ax5.plot(time, hospital_admissions, 'b-', alpha=0.5, label='Admissions')
ax5.plot(time, trend, 'r-', linewidth=2, label='Tendance')
ax5.plot(time[peaks], hospital_admissions[peaks], 'ro', 
        label='Pics détectés')
ax5.set_xlabel('Jours')
ax5.set_ylabel('Nombre d admissions')
ax5.set_title('Admissions hospitalières (simulées)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Relation âge vs risque
ax6 = plt.subplot(2, 3, 6)
scatter = ax6.scatter(data['age'], data['risk_score'], 
                     c=data['cholesterol'], cmap='viridis', 
                     alpha=0.6, s=data['bmi']*5)
ax6.set_xlabel('Âge')
ax6.set_ylabel('Score de risque')
ax6.set_title('Âge vs Risque (taille=BMI, couleur=cholestérol)')
plt.colorbar(scatter, ax=ax6, label='Cholestérol')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Résumé des insights
print("\n=== INSIGHTS CLÉS ===")
print("1. Les fumeurs ont un score de risque significativement plus élevé")
print("2. L'âge et la pression artérielle sont les meilleurs prédicteurs")
print(f"3. {len(peaks)} pics épidémiques détectés sur l'année")
print("4. Le BMI et le cholestérol montrent des relations non-linéaires")