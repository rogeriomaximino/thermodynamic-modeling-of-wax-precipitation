import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
from scipy.optimize import fsolve

# Parâmetros iniciais - mantidos os mesmos
alpha, beta, gamma_p = 3.0, 50.0, 140.0
MM_inicial, MM_final = 142.0, 562.0
P = 10**5  # Pa
R_cal = 1.987  # cal/mol*K
R_joule = 8.314  # J/mol*K
passo = 1

# Vetorização do número de carbonos
n_carb = np.arange(10, 41, passo)
n = n_carb.reshape(1, -1)

# Função de distribuição de massa molar
def distribuicao_massa(MM):
    if MM <= gamma_p:
        return 0
    return ((MM - gamma_p)**(alpha-1) * np.exp(-(MM-gamma_p)/beta)) / (beta**alpha * gamma(alpha))

# Cálculo das frações
intervalos = np.linspace(MM_inicial, MM_final, len(n_carb))
fracoes = np.array([quad(distribuicao_massa, gamma_p, intervalos[0])[0]]+[quad(distribuicao_massa,intervalos[i],intervalos[i+1])[0] for i in range(len(intervalos)-1)])

def MM(n):
  return np.where(n == 10,14*n,14*n+2)

massas_molares=MM(n_carb)
MM=massas_molares.reshape(1,-1)

def Vm(MM):
  return 1.0178*MM + 20.7404

def Tf(MM):
  return 374.5 + 0.02617*MM - (20172/MM)

def Hf(MM):
  return 0.1426*MM*Tf(MM)*4.1868

def Cp(MM):
  return 0.3033*MM - 4.635e-4*MM*Tf(MM)

def Psi(T):
  return np.exp(-Hf(MM)/(R_joule*T)*(1-T/Tf(MM)) - Cp(MM)/R_cal*(1-Tf(MM)/T + np.log(Tf(MM)/T)))

def W(MM):
  return -0.3 + np.exp(-3.06826 + 1.04987*MM**0.2)

def Sg(T):
  asg = 1.0551257
  bsg = 1
  csg = 3.7021308
  dsg = 2.98904640
  esg = 0.10440169
  return asg - bsg*np.exp(csg - dsg*MM**esg)

def Tb(T):
  Tb = np.zeros_like(MM, dtype=float)
  Sg_val = Sg(T)

  limite_menor = MM <= 300
  limite_maior = MM > 300

  # Cálculo para MM <= 300
  Tb[limite_menor] = (3.76587*np.exp(3.7741e-3*MM[limite_menor]+2.98404*Sg_val[limite_menor]-4.25288e-3*MM[limite_menor]*Sg_val[limite_menor])*MM[limite_menor]**0.40167*Sg_val[limite_menor]**(-1.58262))

  # Cálculo para MM > 300
  Tb[limite_maior] = (9.33691*np.exp(1.6514e-4*MM[limite_maior]+1.4103*Sg_val[limite_maior]-7.5152e-4*MM[limite_maior]*Sg_val[limite_maior])*MM[limite_maior]**0.5369*Sg_val[limite_maior]**(-0.7276))

  return Tb

def Tc(MM):
  aRD = 10.6443
  bRD = 0.81067
  cRD = 0.53691
  dRD = -0.00051747
  eRD = -0.5444
  fRD = 3.5995e-4
  Tc = (aRD*(Tb(T)*1.8)**bRD*Sg(T)**cRD*np.exp(dRD*Tb(T)
          * 1.8 + eRD*Sg(T) + fRD*Tb(T)*1.8*Sg(T)))/1.8
  return Tc

def Pc(MM):
  aRD = 6.162e6
  bRD = -0.4844
  cRD = 4.0846
  dRD = -0.004725
  eRD = -4.8014
  fRD = 3.1939e-3
  Pc = (aRD*(Tb(T)*1.8)**bRD*Sg(T)**cRD*np.exp(dRD*Tb(T) *
         1.8 + eRD*Sg(T) + fRD*Tb(T)*1.8*Sg(T)))*6894.75728
  return Pc

def PR_EOS(MM, T):
  Ψ = 0.45724
  Ω = 0.07780
  ε = 1 - np.sqrt(2)
  σ = 1 + np.sqrt(2)
  α = (1+(0.37464+(1.54226*W(MM))-0.26992*(W(MM)**2))
          * (1-((T/Tc(MM))**(1/2))))**2
  a = (Ψ*α*(R_joule*Tc(MM))**2)/Pc(MM)
  b = (Ω*R_joule*Tc(MM))/Pc(MM)
  return a, b, ε, σ

def AB(x):
  a, b, ε, σ = PR_EOS(MM, T)
  B = (b*P)/(R_joule*T)
  A = (a*P)/(R_joule**2*T**2)
  return A, B

def ABmix(x, Ma):
  a, b, ε, σ = PR_EOS(MM, T)
  x = x.flatten()
  bmix = np.sum(x*b)
  amix = np.sum(x*np.dot(Ma, x.T))

  Bmix = (bmix*P)/(R_joule*T)
  Amix = amix*P/(R_joule**2)/(T**2)
  return Amix, Bmix

def Z_PR(A, B):
  coef = [1, B - 1, A - 3*B**2 - 2*B, B*(B**2 + B - A)]
  roots = np.roots(coef)
  Zl = min(r.real for r in roots if r.real > 0)
  return Zl

def vecZ(A, B):
  Zl = []
  for i in range(A.shape[1]):
        Zl.append(Z_PR(A[0, i], B[0, i]))
  return np.array(Zl)

def coefl(Z):
  a, b, ε,σ = PR_EOS(MM,T)
  B = (b*P)/(R_joule*(T))
  A = a*P/(R_joule**2)/(T**2)
  phi = np.exp(Z - 1 - np.log(Z - B) - (A/(2*np.sqrt(2)*B))
                 * np.log((Z + 2.414*B)/(Z - 0.414*B)))
  return phi

def coeflmix(Z, x, Ma):
  Amix, Bmix = ABmix(x, Ma)
  A, B = AB(x)
  a, b, ε,σ = PR_EOS(MM,T)
  bmix = np.sum(x*b)
  Bmix = bmix*P/(R_joule*T)
  x = x.flatten()
  amix = np.dot(x.T, np.dot(Ma, x))
  somaux = np.dot(x.T, Ma)
  phimix = np.exp(B/Bmix*(Z-1)-np.log(Z-Bmix)-Amix/(2*np.sqrt(2)*Bmix)
                  * (2*somaux/amix-B/Bmix)*np.log((Z+2.414*Bmix)/(Z-0.414*Bmix)))
  return phimix

def gammaS(T, s):
  V_molar = Vm(MM)

  # Soma dos volumes
  Soma_V = np.sum(s*V_molar)

  # Calculo de Phi_i
  Phi_sol = (s * V_molar) / Soma_V

  # Calculo do parâmetro de solubilidade delta_i
  Delta_sol = (5.51328 + 1.44476 * np.log(n)) * np.sqrt(4.184)

  # Calculo do parâmetro de solubilidade médio
  Delta_solm = np.sum(Phi_sol * Delta_sol)

  # Calculo do coeficiente de atividade
  coeficientes_atividade = np.exp(
        (V_molar*(Delta_solm-Delta_sol)**2)/(R_joule*T))

  return coeficientes_atividade

def KeqGphi(T, x, s,z):
  a, b,ε,σ = PR_EOS(MM,T)
  Ma = np.sqrt(np.outer(a, a))
  Amix, Bmix = ABmix(x,Ma)
  Zlmix = Z_PR(Amix, Bmix)
  A, B = AB(x)
  Zl = vecZ(A, B)
  KeqGPhi = coeflmix(Zlmix, x, Ma)/coefl(Zl)/gammaS(T,s)/Psi(T)
  return KeqGPhi, Zl

# Paramentros para os cálculos
fracoes_normalizadas = fracoes / np.sum(fracoes)
z = fracoes_normalizadas.reshape(1, -1)
x = z.copy()
s = z.copy()
T_range = np.arange(310, 325, 1)

resultados = {
    "temperatura": [],
    "S_F": [],
    "fases_solidas": {},
    "temperatura_com_s": []
}

# Cálculo
for T in T_range:
    S_F1 = 0
    nmax = 1000  
    tol = 1e-8  
    delta = 1
    nit = 0  
    fobjold = 1e10

    while abs(delta) > tol and nit < nmax:
        K, Zl = KeqGphi(T, x,s,z)
        S_Fold = S_F1

        def fobj_SS(S_F):
            f = (K - 1) / (1 + S_F * (K - 1))
            return np.sum(z * f)

        S_Fcalc = fsolve(fobj_SS, S_F1)[0]

        if S_Fcalc < 0:
            S_F1 = 0
        elif S_Fcalc > 1:
            S_F1 = 1
        else:
            S_F1 = S_Fcalc

        x = z / (1 + S_F1 * (K - 1))
        x = x / np.sum(x)
        s = K * x
        s = s / np.sum(s)
        L_F = 1 - S_F1
        delta = np.abs(S_F1 - S_Fold)
        nit += 1

    resultados["temperatura"].append(T)
    resultados["S_F"].append(S_F1)

    if 0 < S_F1 < 1:
        resultados["temperatura_com_s"].append(T)
        for i, valor in enumerate(s.flatten()):
            carbono = n_carb[i]
            if carbono not in resultados["fases_solidas"]:
                resultados["fases_solidas"][carbono] = []
            resultados["fases_solidas"][carbono].append(valor)

# Gráficos

# Composição da Fase Sólida vs Temperatura
plt.figure(0, figsize=(12, 6))
for carbono, valores in resultados["fases_solidas"].items():
    plt.plot(resultados["temperatura_com_s"], valores, '-', label=f'C{carbono}')

# Calcular número de colunas dinamicamente
n_legendas = len(resultados["fases_solidas"])
n_colunas = max(1, (n_legendas - 1) // 10 + 1)

plt.xlabel("Temperatura (K)")
plt.ylabel("Fração molar da fase Sólida (y)")
plt.title("Composição da Fase Sólida vs Temperatura")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=n_colunas)
plt.grid(True, alpha=0.3)

# Fração Sólida Total vs Temperatura
plt.figure(1, figsize=(10, 6))
plt.plot(resultados["temperatura"], resultados["S_F"], '-', linewidth=2)
plt.xlabel("Temperatura (K)")
plt.ylabel("Fração Massica")
plt.title("Fração Sólida Total vs Temperatura")
plt.grid(True, alpha=0.3)

temp_text = f'Temperatura:\n{T_range[0]}K a {T_range[-1]}K'
plt.text(0.02, 0.18, temp_text, transform=plt.gca().transAxes, 
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

cn_text = f'C: {n_carb[0]} a {n_carb[-1]}\nPasso: {passo}'
plt.text(0.02, 0.10, cn_text, transform=plt.gca().transAxes, 
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()