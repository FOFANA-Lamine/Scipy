

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Exemple 1: Distributions de probabilité
# Distribution normale
mu, sigma = 0, 1
x = np.linspace(-4, 4, 1000)

# PDF et CDF
pdf_norm = stats.norm.pdf(x, mu, sigma)
cdf_norm = stats.norm.cdf(x, mu, sigma)

# Génération d'échantillons aléatoires
samples = stats.norm.rvs(loc=mu, scale=sigma, size=1000)

# Test de normalité (Shapiro-Wilk)
stat_shapiro, p_shapiro = stats.shapiro(samples)
print(f"Test de Shapiro-Wilk: statistique={stat_shapiro:.4f}, p-value={p_shapiro:.4f}")

# Exemple 2: Tests d'hypothèse
# Données d'exemple
group1 = np.random.normal(5, 1.5, 30)
group2 = np.random.normal(6, 1.5, 30)

# Test t pour échantillons indépendants
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"\nTest t: t={t_stat:.4f}, p={p_value:.4f}")

# Test de Mann-Whitney (non-paramétrique)
u_stat, p_mann = stats.mannwhitneyu(group1, group2)
print(f"Test de Mann-Whitney: U={u_stat}, p={p_mann:.4f}")

# Exemple 3: Régression linéaire
x_reg = np.random.rand(50) * 10
y_reg = 2.5 * x_reg + 1.2 + np.random.randn(50) * 2

# Régression avec scipy
slope, intercept, r_value, p_value, std_err = stats.linregress(x_reg, y_reg)
print(f"\nRégression linéaire:")
print(f"Pente: {slope:.4f}")
print(f"Ordonnée à l'origine: {intercept:.4f}")
print(f"R²: {r_value**2:.4f}")
print(f"p-value: {p_value:.4f}")

# Exemple 4: ANOVA
group_a = np.random.normal(10, 2, 20)
group_b = np.random.normal(12, 2, 20)
group_c = np.random.normal(11, 2, 20)

f_stat, p_anova = stats.f_oneway(group_a, group_b, group_c)
print(f"\nANOVA: F={f_stat:.4f}, p={p_anova:.4f}")

# Visualisation
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Distribution normale
axes[0, 0].plot(x, pdf_norm, 'b-', label='PDF')
axes[0, 0].plot(x, cdf_norm, 'r--', label='CDF')
axes[0, 0].hist(samples, bins=30, density=True, alpha=0.5, label='Histogramme')
axes[0, 0].set_title('Distribution Normale')
axes[0, 0].legend()
axes[0, 0].grid(True)

# QQ-plot
stats.probplot(samples, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('QQ-Plot')

# Box plot des groupes
axes[0, 2].boxplot([group1, group2], labels=['Groupe 1', 'Groupe 2'])
axes[0, 2].set_title('Box Plot - Comparaison de groupes')

# Histogrammes comparés
axes[1, 0].hist(group1, bins=15, alpha=0.5, label='Groupe 1', density=True)
axes[1, 0].hist(group2, bins=15, alpha=0.5, label='Groupe 2', density=True)
axes[1, 0].set_title('Distribution des groupes')
axes[1, 0].legend()

# Régression linéaire
axes[1, 1].scatter(x_reg, y_reg, alpha=0.5, label='Données')
x_fit = np.array([x_reg.min(), x_reg.max()])
y_fit = slope * x_fit + intercept
axes[1, 1].plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'y = {slope:.2f}x + {intercept:.2f}')
axes[1, 1].set_title('Régression Linéaire')
axes[1, 1].legend()

# ANOVA - Box plot multiple
axes[1, 2].boxplot([group_a, group_b, group_c], 
                   labels=['Groupe A', 'Groupe B', 'Groupe C'])
axes[1, 2].set_title('ANOVA - Comparaison multiple')

plt.tight_layout()
plt.show()