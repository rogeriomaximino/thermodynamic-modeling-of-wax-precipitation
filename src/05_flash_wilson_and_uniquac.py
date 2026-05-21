import numpy as np
import matplotlib.pyplot as plt

# Dados de Entrada do Artigo (BIM0 - Coutinho, 1999)

n = np.array([10, 18, 19, 20, 21, 22, 23, 24, 25, 26,
              27, 28, 29, 30, 31, 32, 33, 34, 35, 36])

wt = np.array([63.84, 4.276, 3.871, 3.494, 3.149, 2.828, 2.536, 2.270,
               2.028, 1.811, 1.612, 1.441, 1.274, 1.130, 1.004, 0.887,
               0.788, 0.695, 0.522, 0.541])

# Dados de Hf, Tf, Ht,Tt (Broadhurst, 1962)

Hf_kcal = np.array([6.863, 10.313, 10.95, 11.175, 11.40, 11.70, 12.90,
                    13.12, 13.80, 14.22, 14.44, 15.45, 15.80, 16.45,
                    17.375, 18.30, 18.705, 19.11, 20.65, 21.23])

Hf = Hf_kcal * 4184  # J/mol

Tf = np.array([233.6, 300.15, 305.2, 309.3, 313.4, 317.2, 320.7, 323.8,
               326.7, 329.5, 332.0, 334.4, 336.6, 338.6, 340.9, 342.5,
               344.3, 345.9, 347.7, 349.1])

T_tr = np.array([226.9, 289.45, 295.2, 300.45, 305.7, 309.7, 313.7,
                 316.95, 320.2, 323.2, 326.2, 328.8, 331.4, 333.5, 335.7,
                 338.15, 340.6, 342.6, 345.1, 347.0])

H_tr_kcal = np.array([1.639, 2.9725, 3.30, 3.50, 3.70, 4.45, 5.20, 6.715,
                      6.23, 6.575, 6.92, 7.23, 7.54, 7.124, 6.70, 6.70,
                      6.70, 6.70, 7.30, 7.30])

H_tr = H_tr_kcal * 4184  # J/mol

# Dados da curva calculada com Wilson (Artigo)

dados_wilson = np.array([
    [3.0862E+02, 3.9216E-01], [3.0789E+02, 2.3529E+00], [3.0716E+02, 3.5294E+00],
    [3.0634E+02, 5.0980E+00], [3.0525E+02, 6.8627E+00], [3.0434E+02, 9.2157E+00],
    [3.0325E+02, 1.1176E+01], [3.0215E+02, 1.3137E+01], [3.0124E+02, 1.5294E+01],
    [3.0061E+02, 1.6667E+01], [2.9961E+02, 1.8627E+01], [2.9824E+02, 2.1176E+01],
    [2.9706E+02, 2.4118E+01], [2.9569E+02, 2.7255E+01], [2.9432E+02, 3.0392E+01],
    [2.9305E+02, 3.3333E+01], [2.9223E+02, 3.5294E+01], [2.9159E+02, 3.6667E+01],
    [2.9096E+02, 3.8431E+01], [2.9014E+02, 4.0196E+01], [2.8941E+02, 4.1569E+01],
    [2.8822E+02, 4.4510E+01], [2.8686E+02, 4.7647E+01], [2.8549E+02, 5.0588E+01],
    [2.8404E+02, 5.4118E+01], [2.8267E+02, 5.6863E+01], [2.8121E+02, 6.0196E+01],
    [2.7985E+02, 6.3137E+01], [2.7839E+02, 6.6078E+01], [2.7675E+02, 6.9412E+01],
    [2.7520E+02, 7.2353E+01], [2.7293E+02, 7.6275E+01], [2.7129E+02, 7.9020E+01],
    [2.6974E+02, 8.1373E+01], [2.6801E+02, 8.3725E+01], [2.6628E+02, 8.5882E+01],
    [2.6464E+02, 8.7843E+01], [2.6291E+02, 8.9608E+01]
])
T_wilson = dados_wilson[:, 0]
perc_wilson = dados_wilson[:, 1]

# Dados da curva calculada com Uniquac (Artigo)

dados_uniquac = np.array([
    [2.6282E+02, 8.3346E+01], [2.6326E+02, 8.2763E+01], [2.6364E+02, 8.2179E+01],
    [2.6407E+02, 8.1479E+01], [2.6456E+02, 8.0778E+01], [2.6510E+02, 7.9961E+01],
    [2.6565E+02, 7.9027E+01], [2.6624E+02, 7.8093E+01], [2.6695E+02, 7.6693E+01],
    [2.6782E+02, 7.5292E+01], [2.6863E+02, 7.3658E+01], [2.6934E+02, 7.2374E+01],
    [2.7015E+02, 7.0739E+01], [2.7091E+02, 6.9105E+01], [2.7162E+02, 6.7704E+01],
    [2.7233E+02, 6.6304E+01], [2.7319E+02, 6.4553E+01], [2.7412E+02, 6.2451E+01],
    [2.7488E+02, 6.0700E+01], [2.7558E+02, 5.9300E+01], [2.7618E+02, 5.7899E+01],
    [2.7672E+02, 5.6848E+01], [2.7743E+02, 5.5214E+01], [2.7814E+02, 5.3580E+01],
    [2.7873E+02, 5.2296E+01], [2.7944E+02, 5.1012E+01], [2.8009E+02, 4.9377E+01],
    [2.8080E+02, 4.7977E+01], [2.8145E+02, 4.6459E+01], [2.8221E+02, 4.4825E+01],
    [2.8302E+02, 4.3074E+01], [2.8373E+02, 4.1323E+01], [2.8460E+02, 3.9689E+01],
    [2.8530E+02, 3.7821E+01], [2.8623E+02, 3.5837E+01], [2.8688E+02, 3.4086E+01],
    [2.8764E+02, 3.2451E+01], [2.8856E+02, 3.0117E+01], [2.8927E+02, 2.8833E+01],
    [2.9014E+02, 2.6498E+01], [2.9090E+02, 2.4747E+01], [2.9160E+02, 2.3346E+01],
    [2.9204E+02, 2.2529E+01], [2.9236E+02, 2.1829E+01], [2.9269E+02, 2.1128E+01],
    [2.9329E+02, 2.0078E+01], [2.9405E+02, 1.8444E+01], [2.9459E+02, 1.7510E+01],
    [2.9529E+02, 1.5992E+01], [2.9633E+02, 1.4358E+01], [2.9714E+02, 1.2840E+01],
    [2.9795E+02, 1.1440E+01], [2.9877E+02, 1.0272E+01], [2.9942E+02, 9.2218E+00],
    [3.0007E+02, 8.2879E+00], [3.0067E+02, 7.3541E+00], [3.0105E+02, 6.8872E+00],
    [3.0143E+02, 6.1868E+00], [3.0192E+02, 5.7198E+00], [3.0252E+02, 4.7860E+00],
    [3.0317E+02, 4.0856E+00], [3.0366E+02, 3.5019E+00], [3.0398E+02, 3.0350E+00],
    [3.0436E+02, 2.6848E+00], [3.0474E+02, 2.1012E+00], [3.0539E+02, 1.4008E+00],
    [3.0583E+02, 9.3385E-01], [3.0637E+02, 3.5019E-01], [3.0670E+02, 0.0000E+00]
])
T_uniquac = dados_uniquac[:, 0]
perc_uniquac = dados_uniquac[:, 1]

# Propriedades Físicas

def MM_func(n):
    return n * 12.0107 + (2 * n + 2) * 1.00794

MM = MM_func(n)  

def Sg(MM):
    asg = 1.0551257
    bsg = 1.0
    csg = 3.7021308
    dsg = 2.98904640
    esg = 0.10440169
    return asg - bsg * np.exp(csg - dsg * MM**esg)

def Tb(MM):
    Tb = np.zeros_like(MM, dtype=float)
    Sg_val = Sg(MM)
    mask1 = MM <= 300
    mask2 = MM > 300
    if np.any(mask1):
        Tb[mask1] = (3.76587 * np.exp(3.7741e-3 * MM[mask1] + 2.98404 * Sg_val[mask1] - 4.25288e-3 * MM[mask1] * Sg_val[mask1]) * MM[mask1]**0.40167 * Sg_val[mask1]**(-1.58262))
    if np.any(mask2):
        Tb[mask2] = (9.33691 * np.exp(1.6514e-4 * MM[mask2] + 1.4103 * Sg_val[mask2] - 7.5152e-4 * MM[mask2] * Sg_val[mask2]) * MM[mask2]**0.5369 * Sg_val[mask2]**(-0.7276))
    return Tb

def Tc(MM):
    aRD, bRD, cRD = 10.6443, 0.81067, 0.53691
    dRD, eRD, fRD = -0.00051747, -0.5444, 3.5995e-4
    Tb_val = Tb(MM)
    Sg_val = Sg(MM)
    Tc_val = (aRD * (Tb_val * 1.8)**bRD * Sg_val**cRD * np.exp(dRD * Tb_val * 1.8 + eRD * Sg_val + fRD * Tb_val * 1.8 * Sg_val)) / 1.8
    return Tc_val

def V_VdW(n):
    return 10.23 * n

# Correlação GCVOL (Elbro, 1991)

def V_molar_GCVOL(n, T):
    A_CH3 = 18.960
    B_CH3 = 45.58e-3
    A_CH2 = 12.520
    B_CH2 = 12.94e-3
    n_CH3 = 2
    n_CH2 = n - 2
    V = n_CH3 * (A_CH3 + B_CH3 * T) + n_CH2 * (A_CH2 + B_CH2 * T)
    return V

def omega_n(n):
    return 0.0520750 + 0.0448946 * n - 0.000185397 * n**2

# Correlação PERT2 (D.L. Morgan, R. Kobayashi, 1994)

def Hvap_PERT2(T, Tc, omega):
    R = 8.31441
    Tr = T / Tc
    x = 1 - Tr
    b0 = np.array([5.2804, 12.8650, 1.1710, -13.1160, 0.4858, -1.0880])
    b1 = np.array([0.80022, 273.23, 465.08, -638.51, -145.12, 74.049])
    b2 = np.array([7.2543, -346.45, -610.48, 839.89, 160.05, -50.711])
    exp = np.array([0.3333, 0.8333, 1.2083, 1.0, 2.0, 3.0])
    
    Haster = np.zeros_like(T)
    for i in range(len(T)):
        x_exp = x[i]**exp
        Haster_i = (np.sum(b0 * x_exp) + omega[i] * np.sum(b1 * x_exp) + omega[i]**2 * np.sum(b2 * x_exp))
        Haster[i] = Haster_i * R * Tc[i]
    return Haster

def Hsub_Tf(n, MM, Tf, Hf, H_tr):
    Tc_val = Tc(MM)
    omega_val = omega_n(n)
    Hvap = Hvap_PERT2(Tf, Tc_val, omega_val)
    return Hvap + Hf + H_tr

# Modelos de atividade da fase sólida (Wilson e Uniquac)

def gamma_s_wilson(x_s, T, n, MM, Tf, Hf, H_tr):
    R = 8.31441
    Z = 6.0
    
    Hsub = Hsub_Tf(n, MM, Tf, Hf, H_tr)
    lambda_ii = - (2.0 / Z) * (Hsub - R * Tf)

    N = len(n)
    lambda_ij = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if n[i] <= n[j]:
                lambda_ij[i, j] = lambda_ii[i]
            else:
                lambda_ij[i, j] = lambda_ii[j]

    Lambda = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Lambda[i, j] = np.exp(- (lambda_ij[i, j] - lambda_ii[i]) / (R * T))

    if np.sum(x_s) < 1e-10:
        return np.ones(N)

    gamma = np.zeros(N)
    for i in range(N):
        sum1 = np.sum(x_s * Lambda[i, :])
        sum2 = 0.0
        for k in range(N):
            denom = np.sum(x_s * Lambda[k, :])
            if denom > 1e-20:
                sum2 += x_s[k] * Lambda[k, i] / denom
        if sum1 < 1e-20:
            sum1 = 1e-20
        ln_gamma_i = 1 - np.log(sum1) - sum2
        gamma[i] = np.exp(ln_gamma_i)
    return gamma

def gamma_s_uniquac(x_s, T, n, MM, Tf, Hf, H_tr):
    R = 8.31441
    Z = 6.0

    Hsub = Hsub_Tf(n, MM, Tf, Hf, H_tr)
    lambda_ii = - (2.0 / Z) * (Hsub - R * Tf)

    N = len(n)
    lambda_ij = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if n[i] <= n[j]:
                lambda_ij[i, j] = lambda_ii[i]
            else:
                lambda_ij[i, j] = lambda_ii[j]

    R_CH3 = 0.9011
    Q_CH3 = 0.848
    R_CH2 = 0.6744
    Q_CH2 = 0.540

    r_org = np.zeros(N)
    q_org = np.zeros(N)
    for i, nc in enumerate(n):
        n_CH3 = 2
        n_CH2 = max(0, nc - 2)
        r_org[i] = n_CH3 * R_CH3 + n_CH2 * R_CH2
        q_org[i] = n_CH3 * Q_CH3 + n_CH2 * Q_CH2

    r = r_org / 6.744
    q = q_org / 5.40

    if np.sum(x_s) < 1e-10:
        return np.ones(N)

    sum_rx = np.sum(x_s * r)
    sum_qx = np.sum(x_s * q)
    if sum_rx < 1e-20 or sum_qx < 1e-20:
        return np.ones(N)
        
    phi = x_s * r / sum_rx
    theta = x_s * q / sum_qx

    l = (Z / 2) * (r - q) - (r - 1)

    tau_ji = np.zeros((N, N))
    tau_ij = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            tau_ji[j, i] = np.exp(- (lambda_ij[i, j] - lambda_ii[i]) / (q[i] * R * T))
            tau_ij[i, j] = np.exp(- (lambda_ij[i, j] - lambda_ii[j]) / (q[j] * R * T))

    gamma = np.zeros(N)
    for i in range(N):
        ln_gamma_c = np.log(phi[i] / x_s[i]) + (Z / 2) * q[i] * np.log(theta[i] / phi[i]) \
                     + l[i] - (phi[i] / x_s[i]) * np.sum(x_s * l)

        sum_theta_tau_ji = np.sum(theta * tau_ji[:, i])
        if sum_theta_tau_ji < 1e-20:
            sum_theta_tau_ji = 1e-20
        term1 = np.log(sum_theta_tau_ji)

        term2 = 0.0
        for j in range(N):
            sum_theta_tau_kj = np.sum(theta * tau_ij[:, j])
            if sum_theta_tau_kj < 1e-20:
                sum_theta_tau_kj = 1e-20
            term2 += theta[j] * tau_ij[i, j] / sum_theta_tau_kj

        ln_gamma_r = q[i] * (1 - term1 - term2)
        gamma[i] = np.exp(ln_gamma_c + ln_gamma_r)

    return gamma

# Modelo de atividade da fase líquida (Flory Free Volume)

def gamma_l_ffv(x_l, T, n):
    V_i = V_molar_GCVOL(n, T)       
    V_wi = V_VdW(n)                 

    numerador = x_l * (V_i**(1/3) - V_wi**(1/3))**3.3
    denominador = np.sum(numerador)
    
    if denominador < 1e-20:
        return np.ones_like(x_l)
    
    phi_i = numerador / denominador
    
    with np.errstate(divide='ignore', invalid='ignore'):
        term = phi_i / x_l
        term[x_l < 1e-20] = 1.0
        gamma_l = np.exp(np.log(term) + 1 - term)
    return gamma_l

# Rachford Rice

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

# Flash SL

mols = wt / MM
z_molar = mols / np.sum(mols)
z = z_molar

indices_parafinas = np.where(n >= 18)[0]
T_range = np.arange(260, 320, 1)

def run_flash(model='wilson'):
    resultados = {
        "temperatura": [],
        "percent_crystallized": []
    }

    x = z.copy()
    s = z.copy()
    S_F = 0.0

    R = 8.31441
    tol_k = 1e-6

    for T in T_range:
        nit_k = 0
        err_k = 1.0
        
        fusao = (Hf / (R * Tf)) * (Tf / T - 1)
        termo_transicao = np.zeros_like(Hf)
        for i in range(len(Hf)):
            if T < T_tr[i] and T_tr[i] > 0:
                termo_transicao[i] = (H_tr[i] / (R * T_tr[i])) * (T_tr[i] / T - 1)
        K = np.exp(fusao + termo_transicao) 

        while err_k > tol_k and nit_k < 200:
            g_l = gamma_l_ffv(x, T, n)
            if model == 'wilson':
                g_s = gamma_s_wilson(s, T, n, MM, Tf, Hf, H_tr)
            else:
                g_s = gamma_s_uniquac(s, T, n, MM, Tf, Hf, H_tr)
            
            K_new = (g_l / g_s) * np.exp(fusao + termo_transicao)
            K = 0.5 * K + 0.5 * K_new

            S_F_calc = rachford_rice(z, K)
            S_F_calc = np.clip(S_F_calc, 0.0, 1.0)

            if S_F_calc > 1e-10:
                x_new = z / (1 + S_F_calc * (K - 1))
                s_new = K * x_new
                x_new = x_new / np.sum(x_new)
                s_new = s_new / np.sum(s_new)
            else:
                x_new = z.copy()
                s_new = K * z
                s_new = s_new / np.sum(s_new)

            err_k = np.max(np.abs(x - x_new)) + abs(S_F - S_F_calc)
            x = x_new
            s = s_new
            S_F = S_F_calc
            nit_k += 1

        resultados["temperatura"].append(T)
        
        if S_F > 1e-6:
            m_s_i = s * MM * S_F
            massa_crist_C18 = np.sum(m_s_i[indices_parafinas])
            massa_C18_feed = np.sum(z[indices_parafinas] * MM[indices_parafinas])
            percent = (massa_crist_C18 / massa_C18_feed) * 100
        else:
            percent = 0.0
            
        resultados["percent_crystallized"].append(percent)
    
    return resultados

print("Executando flash com modelo Wilson...")
result_wilson = run_flash(model='wilson')
print("Executando flash com modelo UNIQUAC...")
result_uniquac = run_flash(model='uniquac')

# Gráficos

plt.figure(figsize=(12, 7))

plt.plot(result_wilson["temperatura"], result_wilson["percent_crystallized"],
         'b-', linewidth=2, label='Wilson (calculado)')
plt.plot(result_uniquac["temperatura"], result_uniquac["percent_crystallized"],
         'r-', linewidth=2, label='UNIQUAC (calculado)')

plt.scatter(T_wilson, perc_wilson,
            facecolors='none', edgecolors='b',
            s=80, linewidths=1.8, label='Artigo (Wilson)')
plt.scatter(T_uniquac, perc_uniquac,
            facecolors='none', edgecolors='r',
            s=80, linewidths=1.8, label='Artigo (UNIQUAC)')

plt.xlabel('Temperatura (K)')
plt.ylabel('Porcentagem total de parafinas cristalizadas (%)')
plt.title('Comparação Wilson vs UNIQUAC - Cristalização Total (BIM0)')
plt.ylim(0, 90)
plt.xlim(260, 320)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Cálculo dos erros

def calcular_erro(result, T_exp, perc_exp, nome_modelo):
    erros = []
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO COM DADOS DO ARTIGO ({nome_modelo}):")
    print("-"*70)
    print("Temperatura (K) | Artigo (%) | Calculado (%) | Diferença (%)")
    print("-"*70)
    
    for i in range(len(T_exp)):
        if T_exp[i] >= T_range[0] and T_exp[i] <= T_range[-1]:
            idx_calc = np.argmin(np.abs(np.array(result["temperatura"]) - T_exp[i]))
            exp_val = perc_exp[i]
            calc_val = result["percent_crystallized"][idx_calc]
            erro_abs = abs(exp_val - calc_val)
            print(f"{T_exp[i]:.1f} | {exp_val:>15.2f} | {calc_val:>13.2f} | {exp_val - calc_val:>13.2f}")
            erros.append(erro_abs)
    
    if erros:
        mae = np.mean(erros)
        print("="*70)
        print(f"Erro Médio para {nome_modelo}: {mae:.2f}%")
    else:
        print("Nenhum ponto na faixa de temperatura calculada.")

calcular_erro(result_wilson, T_wilson, perc_wilson, "Wilson")
calcular_erro(result_uniquac, T_uniquac, perc_uniquac, "UNIQUAC")