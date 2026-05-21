import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Parâmetros
alpha, beta, gamma_p = 3.0, 50.0, 140.0
MM_inicial, MM_final = 142.0, 562.0
R = 8.314
passo = 10
comprimentos = np.arange(10, 41, passo)
temperaturas = np.arange(300, 341)

# Distribuição da massa molar
def distribuicao_massa(MM):
    if MM <= gamma_p:
        return 0
    return ((MM - gamma_p)**(alpha-1) * np.exp(-(MM-gamma_p)/beta)) / (beta**alpha * gamma(alpha))

# Cálculo das frações normalizadas
intervalos = np.linspace(MM_inicial, MM_final, len(comprimentos))
fracoes = [quad(distribuicao_massa, gamma_p, intervalos[0])[0]] + \
          [quad(distribuicao_massa, intervalos[i], intervalos[i+1])[0] for i in range(len(intervalos)-1)]
fracoes_normalizadas = np.array(fracoes) / sum(fracoes)

# Cálculo de Psi
def calcular_Psi(i, T):
    MM = 14 * i + 2
    Tf = 374.5 + 0.02617*MM - 20172/MM
    DeltaHf = 0.0526 * MM * Tf * 4.1868
    DeltaCp = 0.3033*MM - 4.635e-4*MM*Tf
    termo1 = (DeltaHf/(R*T)) * (1 - T/Tf)
    termo2 = (DeltaCp/R) * (1 - Tf/T + np.log(Tf/T))
    return np.exp(termo1 + termo2)

Psi_valor = np.array([[calcular_Psi(i, T) for T in temperaturas] for i in comprimentos])

# Rachford-Rice
def RR(z, Psi_coluna):
    def rachford_rice_eq(V):
        return np.sum(z * (Psi_coluna - 1) / (1 + V * (Psi_coluna - 1)))
    V = fsolve(rachford_rice_eq, 0.5)[0]
    V = np.clip(V, 0, 1)
    return V

Fase_Liquida = np.zeros((len(comprimentos), len(temperaturas)))
Fase_Solida = np.zeros_like(Fase_Liquida)
V_Valor = np.zeros(len(temperaturas))

for k, T in enumerate(temperaturas):
    V_Valor[k] = RR(fracoes_normalizadas, Psi_valor[:, k])
    
    # Frações Molares de cada fase
    Fase_Liquida[:, k] = fracoes_normalizadas / (1 + V_Valor[k] * (Psi_valor[:, k] - 1))
    Fase_Solida[:, k] = Psi_valor[:, k] * Fase_Liquida[:, k]

# Gráficos

# Fração Molar da Fase Sólida x Temperatura
plt.figure(0, figsize=(12, 6))
for m in range(len(comprimentos)):
    Cn = comprimentos[m]
    plt.plot(temperaturas, Fase_Solida[m, :], '-', label=f'C{Cn}')

# Calcular número de colunas dinamicamente
n_legendas = len(comprimentos)
n_colunas = max(1, (n_legendas - 1) // 10 + 1)

plt.xlabel("Temperatura (K)")
plt.ylabel("Fração molar da fase Sólida (y)")
plt.title(f"Fração molar da fase Sólida em função da Temperatura")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=n_colunas)
plt.grid(True, alpha=0.3)

# Fração Mássica x Temperatura
plt.figure(1, figsize=(10, 6))
plt.plot(temperaturas, V_Valor, '-', linewidth=2)
plt.xlabel("Temperatura (K)")
plt.ylabel("Fração Massica")
plt.title("Fração Massica em função da Temperatura")
plt.grid(True, alpha=0.3)

temp_text = f'Temperatura:\n{temperaturas[0]}K a {temperaturas[-1]}K'
plt.text(0.02, 0.18, temp_text, transform=plt.gca().transAxes, 
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

cn_text = f'C: {comprimentos[0]} a {comprimentos[-1]}\nPasso: {passo}'
plt.text(0.02, 0.10, cn_text, transform=plt.gca().transAxes, 
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()