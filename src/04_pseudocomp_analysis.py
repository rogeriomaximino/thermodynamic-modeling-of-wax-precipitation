import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
from scipy.stats import gamma
from scipy.special import gamma as gamma_func
from scipy.integrate import quad

# Dados
n_c = np.array([10,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
wt = np.array([63.84,4.276,3.871,3.494,3.149,2.828,2.536,2.270,2.028,1.811,
               1.612,1.441,1.274,1.130,1.004,0.887,0.788,0.695,0.522,0.541])

# Cálculo das frações molares
MM = n_c * 12.0107 + (2 * n_c + 2) * 1.00794
frac_molar = (wt / MM) / np.sum(wt / MM)

frac_molar_corrigida = (wt[1:] / MM[1:]) / np.sum(wt[1:] / MM[1:])

# Gráfico Fração Mássica x Molar
plt.plot(n_c, wt/100, 'o-', label='Fração mássica')
plt.plot(n_c, frac_molar, 'o-', label='Fração molar')
plt.legend()
plt.show()

# Ajuste da distribuição Gamma
def gamma_cdf(alpha, beta, shift, n1, n2):
    return gamma.cdf(n2-shift, alpha, scale=beta) - gamma.cdf(n1-shift, alpha, scale=beta)

def objective(params, n, frac_target):
    alpha, beta, shift = params
    error = sum((frac_target[i] - gamma_cdf(alpha, beta, shift, n[i], n[i+1]))**2 
                for i in range(len(n)-1))
    return error

parametros_iniciais = [1,5,10]
result = minimize(objective, parametros_iniciais, args=(n_c[1:], frac_molar_corrigida))
alpha, beta, gamma_p = result.x

n_max = int(gamma.ppf(0.999, alpha, loc=gamma_p, scale=beta))
n_valores = np.arange(n_max + 1)
pdf_valores = gamma.pdf(n_valores, alpha, loc=gamma_p, scale=beta)

print(f"Valor que recupera 99.9% da integral: {n_max}")

# Gráfico da PDF
plt.figure(figsize=(10, 6))
plt.plot(n_valores, pdf_valores, label=f'Gamma PDF (α={alpha:.2f}, β={beta:.2f})')
plt.plot(n_c[1:], frac_molar_corrigida, 'o-', label='Dados corrigidos')
plt.xlim(15, 50)
plt.xlabel('n')
plt.ylabel('Densidade de probabilidade')
plt.legend()
plt.grid(True)
plt.show()

gamma_pdf = lambda x: gamma.pdf(x, alpha, loc=gamma_p, scale=beta)
n_corrigido = n_c[1:]

frac_calculada = np.zeros(len(n_corrigido))

for i in range(len(n_corrigido)):
    frac_calculada[i] = quad(gamma_pdf, n_corrigido[i], n_corrigido[i] + 1)[0]

frac_calculada /= np.sum(frac_calculada)

# Gráfico Valores Calculados x Originais
plt.plot(n_corrigido, frac_calculada, 'o-', label='Valores calculados')
plt.plot(n_corrigido, frac_molar_corrigida, 'o-', label='Valores originais')
plt.legend()
plt.show()

print("VALORES CALCULADOS")
for i in range(len(frac_calculada)):
    intervalo = f"{n_corrigido[i]} até {n_corrigido[i]+1}"
    print(f"frac_calculada[{i}] = {frac_calculada[i]:.8f}  [{intervalo}]")

print(f"\nSoma final (normalizada): {np.sum(frac_calculada):.8f}")


# Parâmetros iniciais
P = 10**5  # Pa
R_cal = 1.987  # cal/mol*K
R_joule = 8.314  # J/mol*K

# Número de pseudocomponentes
n_pseudos = 5
print(f"\nUSANDO {n_pseudos} PSEUDOCOMPONENTES")

n_min = 18
n_max = 36

limites_n = np.linspace(n_min, n_max, n_pseudos + 1)

fracoes_pseudo = []
n_medio_pseudo = []

for i in range(len(limites_n)-1):
    inicio = limites_n[i]
    fim = limites_n[i+1]
    frac_intervalo = quad(gamma_pdf, inicio, fim)[0]
    fracoes_pseudo.append(frac_intervalo)
    n_medio_pseudo.append((inicio + fim) / 2)

fracoes_pseudo = np.array(fracoes_pseudo)
fracoes_pseudo /= np.sum(fracoes_pseudo)

print(f"Número de pseudocomponentes: {len(n_medio_pseudo)}")
print(f"Limites de n: {limites_n}")
print(f"n médios: {n_medio_pseudo}")
print(f"Frações normalizadas: {fracoes_pseudo}")
print(f"Soma das frações: {np.sum(fracoes_pseudo):.8f}")

def MM_func(n):
    return n * 12.0107 + (2 * n + 2) * 1.00794

def Vm(MM):
    return 1.0178*MM + 20.7404

def Tf(MM):
    return 374.5 + 0.02617*MM - (20172/MM)

def Hf(MM):
    return 0.0526*MM*Tf(MM)*4.1868

def Cp(MM):
    return 0.3033*MM - 4.635e-4*MM*Tf(MM)

def Psi(T, MM):
    return np.exp(-Hf(MM)/(R_joule*T)*(1-T/Tf(MM)) - Cp(MM)/R_cal*(1-Tf(MM)/T + np.log(Tf(MM)/T)))

def W(MM):
    return -0.3 + np.exp(-3.06826 + 1.04987*MM**0.2)

def Sg(MM):
    asg = 1.0551257
    bsg = 1
    csg = 3.7021308
    dsg = 2.98904640
    esg = 0.10440169
    return asg - bsg*np.exp(csg - dsg*MM**esg)

def Tb(MM):
    Tb_val = np.zeros_like(MM, dtype=float)
    Sg_val = Sg(MM)

    limite_menor = MM <= 300
    limite_maior = MM > 300

    # Cálculo para MM <= 300
    Tb_val[limite_menor] = (3.76587*np.exp(3.7741e-3*MM[limite_menor]+2.98404*Sg_val[limite_menor]-4.25288e-3*MM[limite_menor]*Sg_val[limite_menor])*MM[limite_menor]**0.40167*Sg_val[limite_menor]**(-1.58262))

    # Cálculo para MM > 300
    Tb_val[limite_maior] = (9.33691*np.exp(1.6514e-4*MM[limite_maior]+1.4103*Sg_val[limite_maior]-7.5152e-4*MM[limite_maior]*Sg_val[limite_maior])*MM[limite_maior]**0.5369*Sg_val[limite_maior]**(-0.7276))

    return Tb_val

def Tc(MM):
    aRD = 10.6443
    bRD = 0.81067
    cRD = 0.53691
    dRD = -0.00051747
    eRD = -0.5444
    fRD = 3.5995e-4
    Tb_val = Tb(MM)
    Sg_val = Sg(MM)
    Tc_val = (aRD*(Tb_val*1.8)**bRD*Sg_val**cRD*np.exp(dRD*Tb_val*1.8 + eRD*Sg_val + fRD*Tb_val*1.8*Sg_val))/1.8
    return Tc_val

def Pc(MM):
    aRD = 6.162e6
    bRD = -0.4844
    cRD = 4.0846
    dRD = -0.004725
    eRD = -4.8014
    fRD = 3.1939e-3
    Tb_val = Tb(MM)
    Sg_val = Sg(MM)
    Pc_val = (aRD*(Tb_val*1.8)**bRD*Sg_val**cRD*np.exp(dRD*Tb_val*1.8 + eRD*Sg_val + fRD*Tb_val*1.8*Sg_val))*6894.75728
    return Pc_val

def PR_EOS(MM, T):
    Ψ = 0.45724
    Ω = 0.07780
    ε = 1 - np.sqrt(2)
    σ = 1 + np.sqrt(2)
    α = (1+(0.37464+(1.54226*W(MM))-0.26992*(W(MM)**2))*(1-((T/Tc(MM))**(1/2))))**2
    a = (Ψ*α*(R_joule*Tc(MM))**2)/Pc(MM)
    b = (Ω*R_joule*Tc(MM))/Pc(MM)
    return a, b, ε, σ

def AB(x, T, MM):
    a, b, ε, σ = PR_EOS(MM, T)
    B = (b*P)/(R_joule*T)
    A = (a*P)/(R_joule**2*T**2)
    return A, B

def ABmix(x, T, Ma):
    a, b, ε, σ = PR_EOS(MM_array, T)
    x = x.flatten()
    bmix = np.sum(x*b)
    amix = np.sum(x*np.dot(Ma, x.T))

    Bmix = (bmix*P)/(R_joule*T)
    Amix = amix*P/(R_joule**2)/(T**2)
    return Amix, Bmix

def Z_PR(A, B):
    coef = [1, B - 1, A - 3*B**2 - 2*B, B*(B**2 + B - A)]
    roots = np.roots(coef)
    real_roots = [r.real for r in roots if r.imag == 0 and r.real > 0]
    if len(real_roots) == 0:
        return 1.0  # fallback
    Zl = min(real_roots)
    return Zl

def vecZ(A, B):
    Zl = []
    for i in range(A.shape[0]):
        Zl.append(Z_PR(A[i], B[i]))
    return np.array(Zl)

def coefl(Z, A, B):
    phi = np.exp(Z - 1 - np.log(Z - B) - (A/(2*np.sqrt(2)*B))*np.log((Z + 2.414*B)/(Z - 0.414*B)))
    return phi

def coeflmix(Z, x, T, Ma):
    Amix, Bmix = ABmix(x, T, Ma)
    A, B = AB(x, T, MM_array)
    a, b, ε, σ = PR_EOS(MM_array, T)
    bmix = np.sum(x*b)
    Bmix = bmix*P/(R_joule*T)
    x = x.flatten()
    amix = np.dot(x.T, np.dot(Ma, x))
    somaux = np.dot(x.T, Ma)
    phimix = np.exp(B/Bmix*(Z-1)-np.log(Z-Bmix)-Amix/(2*np.sqrt(2)*Bmix)*(2*somaux/amix-B/Bmix)*np.log((Z+2.414*Bmix)/(Z-0.414*Bmix)))
    return phimix

def gammaS(T, s, n):
    V_molar = Vm(MM_array)
    Soma_V = np.sum(s*V_molar)
    Phi_sol = (s * V_molar) / Soma_V
    Delta_sol = (5.51328 + 1.44476 * np.log(n)) * np.sqrt(4.184)
    Delta_solm = np.sum(Phi_sol * Delta_sol)
    coeficientes_atividade = np.exp((V_molar*(Delta_solm-Delta_sol)**2)/(R_joule*T))
    return coeficientes_atividade

def KeqGphi(T, x, s, z):
    MM_local = MM_array
    a, b, ε, σ = PR_EOS(MM_local, T)
    Ma = np.sqrt(np.outer(a, a))
    Amix, Bmix = ABmix(x, T, Ma)
    Zlmix = Z_PR(Amix, Bmix)
    A, B = AB(x, T, MM_local)
    Zl = vecZ(A, B)
    phi_mix = coeflmix(Zlmix, x, T, Ma)
    phi_l = coefl(Zl, A, B)
    gamma_s = gammaS(T, s, n_carbono)
    psi_val = Psi(T, MM_local)
    KeqGPhi = phi_mix / phi_l / gamma_s / psi_val
    return KeqGPhi, Zl

n_carbono = np.array(n_medio_pseudo)
fracoes = np.array(fracoes_pseudo)
fracoes_normalizadas = fracoes / np.sum(fracoes)
z = fracoes_normalizadas.reshape(1, -1)
x = z.copy()
s = z.copy()
T_range = np.arange(310, 325, 1)

MM_array = MM_func(n_carbono)

resultados = {
    "temperatura": [],
    "S_F": [],
    "fases_solidas": {},
    "temperatura_com_s": []
}

for T in T_range:
    S_F1 = 0
    nmax = 1000
    tol = 1e-8
    delta = 1
    nit = 0
    fobjold = 1e10

    while abs(delta) > tol and nit < nmax:
        K, Zl = KeqGphi(T, x, s, z)
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
            n_inicio = limites_n[i]
            n_fim = limites_n[i+1]
            carbono_id = f"C{n_inicio:.1f}-{n_fim:.1f}"
            if carbono_id not in resultados["fases_solidas"]:
                resultados["fases_solidas"][carbono_id] = []
            resultados["fases_solidas"][carbono_id].append(valor)

# Gráficos

# Composição da Fase Sólida vs Temperatura
if resultados["temperatura_com_s"]:
    plt.figure(0, figsize=(12, 6))
    for carbono, valores in resultados["fases_solidas"].items():
        plt.plot(resultados["temperatura_com_s"], valores, '-', label=carbono)
    n_legendas = len(resultados["fases_solidas"])
    n_colunas = max(1, (n_legendas - 1) // 10 + 1)

    plt.xlabel("Temperatura (K)")
    plt.ylabel("Fração molar da fase Sólida (y)")
    plt.title(f"Composição da Fase Sólida vs Temperatura ({n_pseudos} pseudos)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=n_colunas)
    plt.grid(True, alpha=0.3)
    plt.show()

# Fração Sólida Total vs Temperatura
plt.figure(1, figsize=(10, 6))
plt.plot(resultados["temperatura"], resultados["S_F"], '-', linewidth=2)
plt.xlabel("Temperatura (K)")
plt.ylabel("Fração Sólida Total (S_F)")
plt.title(f"Fração Sólida Total vs Temperatura ({n_pseudos} pseudos)")
plt.grid(True, alpha=0.3)
temp_text = f'Temperatura:\n{T_range[0]}K a {T_range[-1]}K'
plt.text(0.02, 0.18, temp_text, transform=plt.gca().transAxes, 
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
cn_text = f'Pseudos: {n_pseudos}\nIntervalos: {limites_n[0]:.1f} a {limites_n[-1]:.1f}'
plt.text(0.02, 0.05, cn_text, transform=plt.gca().transAxes, 
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()

print(f"\n INFORMAÇÕES DA DISCRETIZAÇÃO ")
print(f"Número de pseudocomponentes: {n_pseudos}")
print(f"Limites de n: {limites_n}")
print(f"n médios: {n_medio_pseudo}")
print(f"Frações normalizadas: {fracoes_normalizadas}")
print(f"Soma das frações: {np.sum(fracoes_normalizadas):.8f}")
print(f"\n MASSA MOLAR DOS PSEUDOS")
for i, n_medio in enumerate(n_medio_pseudo):
    mm = MM_func(n_medio)
    print(f"Pseudo {i+1}: n médio = {n_medio:.2f}, MM = {mm:.2f} g/mol")