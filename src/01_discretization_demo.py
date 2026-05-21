import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

# Parâmetros
alpha, beta, gamma_p = 3.0, 50.0, 140.0
MM_i, MM_f = 142.0, 562.0

# Função gamma
def f(MM):
    return 0 if MM <= gamma_p else ((MM - gamma_p)**(alpha - 1)) * np.exp(-(MM - gamma_p)/beta) / (beta**alpha * gamma(alpha))

# Cálculo contínuo
MM = np.linspace(MM_i, MM_f, 1000)
cum = [quad(f, MM_i, m)[0] for m in MM]

print("Valores da integral contínua:")
for m, val in zip(MM, cum):
    print(f"MM = {m:.1f}: Probabilidade cumulativa = {val:.6f}")

# Função para gerar os dados discretizados
def discretizar(n):
    edges = np.linspace(MM_i, MM_f, n + 1)
    fracs = [quad(f, edges[i], edges[i+1])[0] for i in range(n)]
    mids = edges[1:]
    cumul = np.cumsum(fracs)
    return [MM_i] + list(mids), [0] + list(cumul), fracs

# Gerar dados discretizados
m30, c30, fr30 = discretizar(30)
m6, c6, fr6 = discretizar(6)
m3, c3, fr3 = discretizar(3)

def print_disc(m, c, fr, name):
    print(f"\nDiscretização {name}:")
    print("MM\tProb. Acumulada\tProb. Intervalo")
    for i in range(len(m)):
        print(f"{m[i]:.1f}\t{c[i]:.6f}\t{fr[i-1]:.6f}" if i else f"{m[i]:.1f}\t{c[i]:.6f}\t{c[i]:.6f}")

print_disc(m30, c30, fr30, "1 em 1 Componente")
print_disc(m6, c6, fr6, "5 em 5 Componentes")
print_disc(m3, c3, fr3, "10 em 10 Componentes")

plt.figure(figsize=(15, 10))

# Grafico geral
plt.subplot(2, 2, 1)
plt.plot(MM, cum, 'k-', lw=2, label='Contínua')
plt.plot(m30, c30, '--o', color='b', ms=4, label='1 em 1')
plt.plot(m6, c6, '-o', color='g', ms=4, label='5 em 5')
plt.plot(m3, c3, '-o', color='r', ms=4, label='10 em 10')
plt.xlabel('Massa Molar (g/mol)')
plt.ylabel('Probabilidade Cumulativa')
plt.title('Comparação Geral')
plt.legend(loc='lower right')
plt.grid(True)

# Grafico de Discretização para pseudocomponentes de 1 em 1
plt.subplot(2, 2, 2)
plt.plot(MM, cum, 'k-', lw=2, label='Contínua')
plt.plot(m30, c30, '-o', color='b', ms=4, label='1 em 1')
plt.xlabel('Massa Molar (g/mol)')
plt.ylabel('Probabilidade Cumulativa')
plt.title('Discretização de 1 em 1 Componente')
plt.legend(loc='lower right')
plt.grid(True)

# Grafico de Discretização para pseudocomponentes de 5 em 5
plt.subplot(2, 2, 3)
plt.plot(MM, cum, 'k-', lw=2, label='Contínua')
plt.plot(m6, c6, '-o', color='g', ms=4, label='5 em 5')
plt.xlabel('Massa Molar (g/mol)')
plt.ylabel('Probabilidade Cumulativa')
plt.title('Discretização de 5 em 5 Componente')
plt.legend(loc='lower right')
plt.grid(True)

# Grafico de Discretização para pseudocomponentes de 10 em 10
plt.subplot(2, 2, 4)
plt.plot(MM, cum, 'k-', lw=2, label='Contínua')
plt.plot(m3, c3, '-o', color='r', ms=4, label='10 em 10')
plt.xlabel('Massa Molar (g/mol)')
plt.ylabel('Probabilidade Cumulativa')
plt.title('Discretização de 10 em 10 Componente')
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.show()