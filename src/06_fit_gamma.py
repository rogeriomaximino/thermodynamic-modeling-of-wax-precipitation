import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gamma
from scipy.integrate import quad
import warnings

warnings.filterwarnings('ignore')

# Dados de entrada (Dauphin 1999)
n_i = np.array([10, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 33, 34, 35, 36])

Hf_i = np.array([6.863, 10.313, 10.95, 11.175, 11.40, 11.70, 12.90,
                 13.12, 13.80, 14.22, 14.44, 15.45, 15.80, 16.45,
                 17.375, 18.30, 18.705, 19.11, 20.65, 21.23]) * 4184
Tf_i = np.array([233.6, 300.15, 305.2, 309.3, 313.4, 317.2, 320.7, 323.8,
                 326.7, 329.5, 332.0, 334.4, 336.6, 338.6, 340.9, 342.5,
                 344.3, 345.9, 347.7, 349.1])
H_tr_i = np.array([1.639, 2.9725, 3.30, 3.50, 3.70, 4.45, 5.20, 6.715,
                   6.23, 6.575, 6.92, 7.23, 7.54, 7.124, 6.70, 6.70,
                   6.70, 6.70, 7.30, 7.30]) * 4184
T_tr_i = np.array([226.9, 289.45, 295.2, 300.45, 305.7, 309.7, 313.7,
                   316.95, 320.2, 323.2, 326.2, 328.8, 331.4, 333.5, 335.7,
                   338.15, 340.6, 342.6, 345.1, 347.0])

T_art = np.array([262.86, 267.74, 272.88, 278.01, 283.14, 288.11, 293.41, 297.95, 303.09, 308.72])
crist_art = np.array([81.000, 73.941, 64.235, 54.000, 43.059, 32.647, 21.353, 14.647, 6.7059, 0.17647])

wt_solv = 63.84
wt_paraf_total = 36.16

# Funções termo-fisicas
def MM_func(n):
    return n * 12.0107 + (2 * n + 2) * 1.00794

def Sg(MM):
    return 1.0551257 - np.exp(3.7021308 - 2.98904640 * MM**0.10440169)

def Tb(MM):
    Tb_v = np.zeros_like(MM, dtype=float)
    Sg_v = Sg(MM)
    m1 = (MM <= 300)
    m2 = (MM > 300)
    if np.any(m1):
        Tb_v[m1] = (3.76587 *
                    np.exp(3.7741e-3 * MM[m1] + 2.98404 * Sg_v[m1]
                           - 4.25288e-3 * MM[m1] * Sg_v[m1]) *
                    MM[m1]**0.40167 * Sg_v[m1]**(-1.58262))
    if np.any(m2):
        Tb_v[m2] = (9.33691 *
                    np.exp(1.6514e-4 * MM[m2] + 1.4103 * Sg_v[m2]
                           - 7.5152e-4 * MM[m2] * Sg_v[m2]) *
                    MM[m2]**0.5369 * Sg_v[m2]**(-0.7276))
    return Tb_v

def Tc(MM):
    Tb_v = Tb(MM)
    Sg_v = Sg(MM)
    return ((10.6443 * (Tb_v * 1.8)**0.81067 * Sg_v**0.53691 *
             np.exp(-0.00051747 * Tb_v * 1.8 - 0.5444 * Sg_v
                    + 3.5995e-4 * Tb_v * 1.8 * Sg_v)) / 1.8)

def omega_n(n):
    return 0.0520750 + 0.0448946 * n - 0.000185397 * n**2

def Hvap_PERT2(T, Tc_val, omega):
    R = 8.31441
    x = 1 - (T / Tc_val)
    b0 = np.array([5.2804, 12.8650, 1.1710, -13.1160, 0.4858, -1.0880])
    b1 = np.array([0.80022, 273.23, 465.08, -638.51, -145.12, 74.049])
    b2 = np.array([7.2543, -346.45, -610.48, 839.89, 160.05, -50.711])
    exp_pow = np.array([1/3, 5/6, 29/24, 1.0, 2.0, 3.0])
    Hvp = np.zeros_like(T)
    for i in range(len(T)):
        x_exp = np.maximum(x[i], 1e-10)**exp_pow
        Hvp_i = (np.sum(b0 * x_exp)
                 + omega[i] * np.sum(b1 * x_exp)
                 + omega[i]**2 * np.sum(b2 * x_exp))
        Hvp[i] = Hvp_i * R * Tc_val[i]
    return Hvp

def Hsub_Tf(n, MM, Tf, Hf, H_tr):
    return Hvap_PERT2(Tf, Tc(MM), omega_n(n)) + Hf + H_tr

def V_molar_GCVOL(n, T):
    n_CH3 = 2.0
    n_CH2 = np.maximum(0.0, n - 2.0)
    return (n_CH3 * (18.960 + 45.58e-3 * T)
            + n_CH2 * (12.520 + 12.94e-3 * T))

def gamma_l_ffv(x_l, T, n):
    V_i = V_molar_GCVOL(n, T)
    V_wi = 10.23 * n
    numer = x_l * (V_i**(1/3) - V_wi**(1/3))**3.3
    denom = np.sum(numer)
    if denom < 1e-20: 
        return np.ones_like(x_l)
    phi_i = numer / denom
    with np.errstate(divide='ignore', invalid='ignore'):
        term = phi_i / x_l
        term[x_l < 1e-20] = 1.0
        return np.exp(np.log(term) + 1 - term)

def gamma_s_wilson(x_s, T, n, MM, Tf, Hf, H_tr):
    R, Z = 8.31441, 6.0
    H_sub = Hsub_Tf(n, MM, Tf, Hf, H_tr)
    lambda_ii = -(2.0 / Z) * (H_sub - R * Tf)
    N = len(n)
    lambda_ij = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            lambda_ij[i, j] = lambda_ii[i] if n[i] <= n[j] else lambda_ii[j]
    Lambda = np.exp(- (lambda_ij - lambda_ii[:, None]) / (R * T))
    if np.sum(x_s) < 1e-10: 
        return np.ones(N)
    gamma_val = np.zeros(N)
    for i in range(N):
        sum1 = np.sum(x_s * Lambda[i, :])
        sum2 = np.sum([x_s[k] * Lambda[k, i] / np.sum(x_s * Lambda[k, :])
                       for k in range(N) if np.sum(x_s * Lambda[k, :]) > 1e-20])
        if sum1 < 1e-20: sum1 = 1e-20
        gamma_val[i] = np.exp(1 - np.log(sum1) - sum2)
    return gamma_val

def rachford_rice(z, K, tol=1e-10, max_iter=100):
    def f(beta):
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(np.isfinite(z*(K-1)/(1+beta*(K-1))),
                            z*(K-1)/(1+beta*(K-1)), 0.0)
        return np.sum(term)
    f0, f1 = f(0.0), f(1.0)
    if f0 <= 0.0: 
        return 0.0
    if f1 >= 0.0: 
        return 1.0
    a, b = 0.0, 1.0
    for _ in range(max_iter):
        beta = 0.5*(a + b)
        fmid = f(beta)
        if abs(fmid) < tol:
            return beta
        if f(a)*fmid < 0:
            b = beta
        else:
            a = beta
    return 0.5*(a + b)

# Flash Sólido-Líquido
def Flash_SL(alpha, beta, gamma_p, T_range, n_pseudos=19):
    n_min = 18
    n_max_calc = int(gamma.ppf(0.95, alpha, loc=gamma_p, scale=beta))
    n_max = max(n_min + 5, min(n_max_calc, 80))
    n_max = max(n_max, 36)
    limites_n = np.linspace(n_min, n_max, n_pseudos + 1)
    
    fracoes_molar_pseudo = []
    n_medio_pseudo = []
    for i in range(n_pseudos):
        inicio = limites_n[i]
        fim = limites_n[i+1]
        frac_intervalo = quad(lambda x: gamma.pdf(x, alpha, loc=gamma_p, scale=beta),
                              inicio, fim)[0]
        num = quad(lambda x: x * gamma.pdf(x, alpha, loc=gamma_p, scale=beta),
                   inicio, fim)[0]
        fracoes_molar_pseudo.append(frac_intervalo)
        n_medio_pseudo.append(num / frac_intervalo if frac_intervalo > 1e-10 else (inicio+fim)/2)
    
    fracoes_molar_pseudo = np.array(fracoes_molar_pseudo)
    
    if np.sum(fracoes_molar_pseudo) > 0:
        fracoes_molar_pseudo /= np.sum(fracoes_molar_pseudo)
    n_medio_pseudo = np.array(n_medio_pseudo)
    
    MM_pseudo = MM_func(n_medio_pseudo)
    MM_media_paraf = np.sum(fracoes_molar_pseudo * MM_pseudo)
    mols_paraf_total = wt_paraf_total / MM_media_paraf if MM_media_paraf > 0 else 0
    mols_solv = wt_solv / MM_func(10.0)
    mols_totais = mols_solv + mols_paraf_total
    z_solv = mols_solv / mols_totais
    z_paraf_global = mols_paraf_total / mols_totais
    z_global = np.insert(fracoes_molar_pseudo * z_paraf_global, 0, z_solv)
    n_global = np.insert(n_medio_pseudo, 0, 10.0)
    MM_global = MM_func(n_global)

    Hf_global = np.interp(n_global, n_i, Hf_i)
    Tf_global = np.interp(n_global, n_i, Tf_i)
    H_tr_global = np.interp(n_global, n_i, H_tr_i)
    T_tr_global = np.interp(n_global, n_i, T_tr_i)
    
    resultados = {"temperatura": [], "percent_cristal": []}
    x = z_global.copy()
    s = z_global.copy()
    S_F = 0.0
    R = 8.31441

    for T in T_range:
        err_k, nit_k = 1.0, 0
        fusao = (Hf_global / (R * Tf_global)) * (Tf_global / T - 1)
        termo_transicao = np.where((T < T_tr_global) & (T_tr_global > 0),
                                   (H_tr_global / (R * T_tr_global)) * (T_tr_global / T - 1),
                                   0.0)
        K = np.exp(fusao + termo_transicao)

        while err_k > 1e-6 and nit_k < 200:
            g_l = gamma_l_ffv(x, T, n_global)
            g_s = gamma_s_wilson(s, T, n_global, MM_global, Tf_global, Hf_global, H_tr_global)
            K_new = (g_l / g_s) * np.exp(fusao + termo_transicao)
            K = 0.5 * K + 0.5 * K_new
            S_F_calc = np.clip(rachford_rice(z_global, K), 0.0, 1.0)

            if S_F_calc > 1e-10:
                x_new = z_global / (1 + S_F_calc * (K - 1))
                s_new = K * x_new
                x_new /= np.sum(x_new)
                s_new /= np.sum(s_new)
            else:
                x_new = z_global.copy()
                s_new = K * z_global
                if np.sum(s_new) > 0:
                    s_new /= np.sum(s_new)

            err_k = np.max(np.abs(x - x_new)) + abs(S_F - S_F_calc)
            x, s, S_F = x_new, s_new, S_F_calc
            nit_k += 1

        resultados["temperatura"].append(T)
        if S_F > 1e-6:
            massa_solida = s * MM_global * S_F
            massa_crist_paraf = np.sum(massa_solida[1:])
            massa_paraf_feed = np.sum(z_global[1:] * MM_global[1:])
            percent = (massa_crist_paraf / massa_paraf_feed) * 100 if massa_paraf_feed > 0 else 0
        else:
            percent = 0.0
        resultados["percent_cristal"].append(percent)

    return resultados

# Otimização
def objective_crist(params):
    alpha, beta, gamma_p = params
    if alpha < 0.1 or beta < 0.1 or gamma_p < 10 or gamma_p > 30:
        return 1e6
    try:
        res = Flash_SL(alpha, beta, gamma_p, T_art)
        crist_calc = np.array(res["percent_cristal"])
        erro = np.sum((crist_calc - crist_art)**2)
        return erro
    except Exception:
        return 1e6

print("Executando o Modelo")
res_opt = minimize(objective_crist, [1.0, 5.0, 16.0], method='Nelder-Mead', options={'maxiter': 200})
alpha_opt, beta_opt, gamma_p_opt = res_opt.x
print(f"Parâmetros otimizados: α={alpha_opt:.3f}, β={beta_opt:.3f}, γ_p={gamma_p_opt:.3f}")

resultados_finais = Flash_SL(alpha_opt, beta_opt, gamma_p_opt, T_art)
cristal_calc = np.array(resultados_finais["percent_cristal"])

print("\nComparação Modelo × Artigo:")
print(" T(K)   Artigo(%)  Modelo(%)")
for T_val, exp_val, mod_val in zip(T_art, crist_art, cristal_calc):
    print(f"{T_val:6.2f}  {exp_val:6.2f}  {mod_val:7.2f}")

# %Crist x T (Modelo vs Artigo)
plt.figure(figsize=(8,6))
T_plot = np.linspace(min(T_art)-5, max(T_art)+5, 100)
resultados_curva = Flash_SL(alpha_opt, beta_opt, gamma_p_opt, T_plot)
cristal_plot = np.array(resultados_curva["percent_cristal"])
plt.plot(T_plot, cristal_plot, 'b-', linewidth=2, label='Modelo')
plt.scatter(T_art, crist_art, color='red', zorder=5, label='Dados Artigo')
plt.xlabel('Temperatura (K)')
plt.ylabel('Porcentagem Cristalizada (%)')
plt.title('Modelo vs Artigo')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Frac Mássica x n (Modelo vs Artigo)
n_int = np.arange(18, 51)
frac_molar = np.zeros_like(n_int, dtype=float)
for i, n in enumerate(n_int):
    frac_molar[i] = (gamma.cdf(n+1, alpha_opt, loc=gamma_p_opt, scale=beta_opt) -
                     gamma.cdf(n, alpha_opt, loc=gamma_p_opt, scale=beta_opt))
frac_molar /= np.sum(frac_molar)
MM_int = MM_func(n_int)
massa_modelo = frac_molar * MM_int
frac_mass_modelo = massa_modelo / np.sum(massa_modelo) * 100

# Composição real do artigo (C18-C36)
wt_paraf_real = np.array([4.276, 3.871, 3.494, 3.149, 2.828, 2.536, 2.270,
                          2.028, 1.811, 1.612, 1.441, 1.274, 1.130, 1.004,
                          0.887, 0.788, 0.695, 0.522, 0.541])
frac_mass_real = wt_paraf_real / np.sum(wt_paraf_real) * 100
n_real = np.arange(18, 18+len(frac_mass_real))

plt.figure(figsize=(8,6))
plt.plot(n_int, frac_mass_modelo, 'b-', linewidth=2, label='Modelo')
plt.scatter(n_real, frac_mass_real, color='red', zorder=5, label='Dados Artigo')
plt.xlabel('Número de Carbonos')
plt.ylabel('Fração Mássica (%)')
plt.title('Modelo vs Artigo')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()