# -*- coding: utf-8 -*-
"""
==============================================================================
 SIMULAÇÃO TÉRMICA 2D — Intel Core i7-13700K (Raptor Lake-S)
 Disciplina: Materiais e Dispositivos Elétricos — Engenharia Física
 Padrão Visual e Tipográfico: ABNT NBR 14724 / Normas de Apresentação IBGE
 Otimização Numérica: Solver FDM com Numba JIT e Duplo Buffer (Ping-Pong)
==============================================================================
"""
import sys
import io
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend não-bloqueante de alta precisão
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from numba import njit
import warnings
warnings.filterwarnings("ignore")

# Força UTF-8 no console do Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# 1. CONSTANTES FUNDAMENTAIS E SEMICONDUTORAS (Silício Monocristalino)
# =============================================================================
Q_E       = 1.602176634e-19   # Carga elementar [C]
K_B_J     = 1.380649e-23      # Constante de Boltzmann [J/K]
K_B_EV    = 8.617333262e-5    # Constante de Boltzmann [eV/K]
HBAR      = 1.054571817e-34   # Constante reduzida de Planck [J·s]
M_0       = 9.1093837e-31     # Massa de repouso do elétron [kg]

# --- Silício: Massas efetivas de densidade de estados (primeiros princípios) ---
# Banda de condução: 6 vales elipsoidais ao longo de <100> (ml=0.98, mt=0.19)
# mc* = 6^(2/3) * (ml * mt^2)^(1/3) ≈ 1.08 * m0
M_C_EFF   = 1.08 * M_0        # Massa efetiva dos elétrons [kg]
# Banda de valência: buracos pesados e leves degenerados no ponto Γ
# mv* = (mhh^(3/2) + mlh^(3/2))^(2/3) ≈ 0.56 * m0
M_V_EFF   = 0.56 * M_0        # Massa efetiva das lacunas [kg]

# --- Silício: Propriedades térmicas e de transporte ---
RHO_SI    = 2329.0            # Densidade volumétrica do Si [kg/m³]
CP_SI_300 = 700.0             # Calor específico a 300 K [J/(kg·K)]
K_SI_300  = 148.0             # Condutividade térmica a 300 K [W/(m·K)]
EG0_SI    = 1.1695            # Bandgap do Si em T=0 K [eV] (Varshni)
ALPHA_V   = 4.73e-4           # Coeficiente α de Varshni [eV/K]
BETA_V    = 636.0             # Temperatura característica β de Varshni [K]
MU_N_300  = 1417.0            # Mobilidade de elétrons a 300 K [cm²/(V·s)]
MU_P_300  = 470.5             # Mobilidade de lacunas a 300 K [cm²/(V·s)]
GAMMA_N   = 2.42              # Expoente de fônons para elétrons (Jacoboni-Canali)
GAMMA_P   = 2.20              # Expoente de fônons para lacunas
N_D       = 1e17              # Concentração de dopantes doadores (tipo-n) [cm⁻³]
TAU_N     = 1.0e-6            # Tempo de vida de recombinação de elétrons [s]
TAU_P     = 1.0e-6            # Tempo de vida de recombinação de lacunas [s]

# =============================================================================
# 2. MODELOS FÍSICOS ANALÍTICOS (Física dos Semicondutores da Disciplina)
# =============================================================================
def Nc_states(T):
    """
    Densidade efetiva de estados na Banda de Condução [cm⁻³]:
    Nc(T) = 2 · [ (mc* · kB · T) / (2π · ħ²) ]^(3/2)
    """
    T_s = np.maximum(T, 1.0)
    val_m3 = 2.0 * ((M_C_EFF * K_B_J * T_s) / (2.0 * np.pi * HBAR**2))**1.5
    return val_m3 * 1.0e-6  # converte m⁻³ para cm⁻³

def Nv_states(T):
    """
    Densidade efetiva de estados na Banda de Valência [cm⁻³]:
    Nv(T) = 2 · [ (mv* · kB · T) / (2π · ħ²) ]^(3/2)
    """
    T_s = np.maximum(T, 1.0)
    val_m3 = 2.0 * ((M_V_EFF * K_B_J * T_s) / (2.0 * np.pi * HBAR**2))**1.5
    return val_m3 * 1.0e-6  # converte m⁻³ para cm⁻³

def Eg(T):
    """Bandgap de Varshni: Eg(T) = Eg(0) - α·T² / (T + β) [eV]"""
    T_s = np.maximum(T, 1.0)
    return EG0_SI - ALPHA_V * T_s**2 / (T_s + BETA_V)

def ni(T):
    """
    Concentração intrínseca de portadores [cm⁻³]:
    ni(T) = √(Nc · Nv) · exp(-Eg / (2 · kB · T))
    """
    T_s = np.maximum(T, 1.0)
    Nc = Nc_states(T_s)
    Nv = Nv_states(T_s)
    return np.sqrt(Nc * Nv) * np.exp(-Eg(T_s) / (2.0 * K_B_EV * T_s))

def carrier_concentrations(T, Nd=N_D):
    """
    Neutralidade de carga (Si tipo-n): no = Nd/2 + √((Nd/2)² + ni²) [cm⁻³]
    Lei de Ação das Massas: po = ni² / no [cm⁻³] (no · po = ni²)
    """
    ni_T = ni(T)
    n_o = 0.5 * Nd + np.sqrt((0.5 * Nd)**2 + ni_T**2)
    p_o = ni_T**2 / n_o
    return n_o, p_o

def fermi_levels(T, Nd=N_D, Ev=0.0):
    """
    Nível de Fermi e Nível Intrínseco [eV]:
    Ei = (Ec + Ev)/2 + (3/4) · kB·T · ln(mv* / mc*)
    EF = Ei + kB·T · ln(no / ni) = Ec - kB·T · ln(Nc / no)
    """
    T_s = np.maximum(T, 1.0)
    Eg_T = Eg(T_s)
    Ec = Ev + Eg_T
    ni_T = ni(T_s)
    n_o, _ = carrier_concentrations(T_s, Nd)
    shift_mass = 0.75 * K_B_EV * T_s * np.log(M_V_EFF / M_C_EFF)
    Ei = 0.5 * (Ec + Ev) + shift_mass
    EF = Ei + K_B_EV * T_s * np.log(np.maximum(n_o / ni_T, 1e-300))
    return Ec, Ev, Ei, EF, EF - Ei

def mu_n(T):
    """Mobilidade de elétrons (Jacoboni-Canali): μ_n(T) = μ_n(300) · (T/300)^(-2.42)"""
    return MU_N_300 * (np.maximum(T, 1.0) / 300.0)**(-GAMMA_N)

def mu_p(T):
    """Mobilidade de lacunas: μ_p(T) = μ_p(300) · (T/300)^(-2.20)"""
    return MU_P_300 * (np.maximum(T, 1.0) / 300.0)**(-GAMMA_P)

def relaxation_times(T):
    """Tempo de relaxação de momento (Drude-Boltzmann): τ = m* · μ / e [s]"""
    mu_n_si = mu_n(T) * 1.0e-4  # cm²/(V·s) -> m²/(V·s)
    mu_p_si = mu_p(T) * 1.0e-4
    tau_e = (M_C_EFF * mu_n_si) / Q_E
    tau_h = (M_V_EFF * mu_p_si) / Q_E
    return tau_e, tau_h

def einstein_diffusion(T):
    """
    Relação de Einstein: D / μ = kB·T / e [cm²/s]
    Dn = μn · (kB·T / e), Dp = μp · (kB·T / e)
    """
    T_s = np.maximum(T, 1.0)
    Vt = K_B_EV * T_s  # Tensão térmica kB·T/e [V]
    return mu_n(T_s) * Vt, mu_p(T_s) * Vt

def diffusion_lengths(T, tau_n=TAU_N, tau_p=TAU_P):
    """Comprimento de difusão de portadores: Ln = √(Dn · τn), Lp = √(Dp · τp) [cm]"""
    Dn, Dp = einstein_diffusion(T)
    return np.sqrt(Dn * tau_n), np.sqrt(Dp * tau_p)

def conductivity(T, Nd=N_D):
    """Condutividade elétrica total: σ(T) = q · (n_o·μ_n + p_o·μ_p) [S/cm]"""
    n_o, p_o = carrier_concentrations(T, Nd)
    return Q_E * (n_o * mu_n(T) + p_o * mu_p(T))

def drift_current_density(T, E_field_V_cm, Nd=N_D):
    """Densidade de corrente de deriva: J_deriva = σ(T) · E [A/cm²]"""
    return conductivity(T, Nd) * E_field_V_cm

# =============================================================================
# 3. GEOMETRIA DO DIE E ARQUITETURA DE POTÊNCIA (i7-13700K Gaming Load ~125 W)
# =============================================================================
NX, NY  = 120, 60             # Dimensões da malha discreta (120 x 60)
DX = DY = 200e-6              # Resolução espacial 200 µm -> Die de 24,0 mm x 12,0 mm
DZ      = 500e-6              # Espessura do substrato de silício (500 µm)
T_AMB   = 300.0               # Temperatura ambiente inicial (26,85 °C / 300 K)

GAMING_POWER = {
    "P-Core": 82.0, "L3-Cache": 18.0, "E-Cluster": 14.0, "IMC-IO": 9.0, "iGPU": 2.0
}
TOTAL_POWER = sum(GAMING_POWER.values())

# Floorplan funcional conciso (name, domain, x0, x1, y0, y1)
BLOCKS = (
    [(f"P-Core {c}", "P-Core", 2 + c*15, 16 + c*15, 42, 57) for c in range(4)] +
    [(f"P-Core {c + 4}", "P-Core", 2 + c*15, 16 + c*15, 26, 41) for c in range(4)] +
    [("E-Cluster 0 (4c)", "E-Cluster", 62, 85, 35, 57),
     ("E-Cluster 1 (4c)", "E-Cluster", 86, 109, 35, 57),
     ("Ring Bus Interconnect", "L3-Cache", 2, 109, 20, 25),
     ("L3 Smart Cache (30 MB)", "L3-Cache", 2, 109, 8, 19),
     ("Mem Ctrl (DDR5)", "IMC-IO", 2, 28, 2, 7),
     ("PCIe 5.0 Ctrl", "IMC-IO", 29, 60, 2, 7),
     ("System Agent", "IMC-IO", 61, 85, 2, 7),
     ("iGPU (UHD 770)", "iGPU", 86, 109, 2, 34)]
)

def build_power_map():
    """Gera matriz de densidade volumétrica Q_gen [W/m³] por fatiamento direto."""
    P_vol = np.zeros((NX, NY), dtype=np.float64)
    cell_vol = DX * DY * DZ
    area_dom = {}
    for _, dom, x0, x1, y0, y1 in BLOCKS:
        area_dom[dom] = area_dom.get(dom, 0) + (x1 - x0) * (y1 - y0)
    for _, dom, x0, x1, y0, y1 in BLOCKS:
        P_vol[x0:x1, y0:y1] = (GAMING_POWER[dom] / area_dom[dom]) / cell_vol
    return P_vol

# =============================================================================
# 4. SOLVER FDM 2D PARALELO COM NUMBA (DOUBLE BUFFERING ZERO-ALLOCATION)
# =============================================================================
alpha_si = K_SI_300 / (RHO_SI * CP_SI_300)
DT_FDM   = (DX**2 / (4.0 * alpha_si)) * 0.40  # Critério de Von Neumann (r = 0.10 <= 0.25)
CK       = DT_FDM * K_SI_300 / (RHO_SI * CP_SI_300 * DX**2)
CQ       = DT_FDM / (RHO_SI * CP_SI_300)

@njit(fastmath=True)
def _fdm_step(T_src, T_dst, P_gen, ck, cq):
    """Executa um passo temporal FDM 2D explícito: ρ·cp·∂T/∂t = k·∇²T + Q_gen"""
    nx, ny = T_src.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            lap = T_src[i+1, j] + T_src[i-1, j] + T_src[i, j+1] + T_src[i, j-1] - 4.0 * T_src[i, j]
            T_dst[i, j] = T_src[i, j] + ck * lap + cq * P_gen[i, j]
    # Condições de contorno adiabáticas (Neumann zero-flux dT/dn = 0 nas 4 bordas)
    for j in range(ny):
        T_dst[0, j] = T_dst[1, j]
        T_dst[nx - 1, j] = T_dst[nx - 2, j]
    for i in range(nx):
        T_dst[i, 0] = T_dst[i, 1]
        T_dst[i, ny - 1] = T_dst[i, ny - 2]

@njit(fastmath=True)
def fdm_batch_kernel(T0, T1, P_gen, ck, cq, n_steps):
    """Executa n_steps em ping-pong double buffering (ZERO alocações de memória)."""
    for _ in range(n_steps // 2):
        _fdm_step(T0, T1, P_gen, ck, cq)
        _fdm_step(T1, T0, P_gen, ck, cq)

# =============================================================================
# 5. EXECUÇÃO DA SIMULAÇÃO TÉRMICA E TRANSIENTE
# =============================================================================
print("=" * 72)
print("  SIMULAÇÃO TÉRMICA 2D — Intel Core i7-13700K (Raptor Lake-S)")
print("  Disciplina: Materiais e Dispositivos Elétricos — Engenharia Física")
print("  Normas de Publicação: ABNT NBR 14724 / Regras de Apresentação IBGE")
print("=" * 72)

# Cálculo analítico da temperatura crítica de inversão intrínseca (ni = Nd)
T_fail_K = brentq(lambda T: float(ni(T)) - N_D, 400, 1500)
T_fail_C = T_fail_K - 273.15
print(f"  Ponto de Colapso Lógico (n_i = N_d): {T_fail_K:.1f} K ({T_fail_C:.1f} °C)")
print(f"  Potência Dissipada em Regime Gaming: {TOTAL_POWER:.0f} W (sem dissipador)")

P_gen_vol = build_power_map()
T0 = np.full((NX, NY), T_AMB, dtype=np.float64)
T1 = np.full((NX, NY), T_AMB, dtype=np.float64)

# Compilação JIT inicial (warmup do kernel C nativo)
print("  Compilando kernel FDM em código de máquina nativo (LLVM)...", end="", flush=True)
t_w = time.perf_counter()
fdm_batch_kernel(T0, T1, P_gen_vol, CK, CQ, 2)
T0.fill(T_AMB)
T1.fill(T_AMB)
print(f" Concluído em {time.perf_counter() - t_w:.2f} s")

N_BATCH = 100
MAX_BATCHES = 10000

hist_t_ms, hist_Tmax_C, hist_ni_ratio, hist_mu_n, hist_sigma = [], [], [], [], []
elapsed_s = 0.0

print("  Executando integração temporal com double buffering...", end="", flush=True)
t_start = time.perf_counter()
for batch in range(MAX_BATCHES):
    fdm_batch_kernel(T0, T1, P_gen_vol, CK, CQ, N_BATCH)
    elapsed_s += N_BATCH * DT_FDM

    T_max = float(np.max(T0))
    ni_T = float(ni(T_max))
    
    hist_t_ms.append(elapsed_s * 1e3)
    hist_Tmax_C.append(T_max - 273.15)
    hist_ni_ratio.append(max(ni_T / N_D, 1e-15))
    hist_mu_n.append(float(mu_n(T_max)))
    hist_sigma.append(float(conductivity(T_max)))

    if ni_T >= N_D or T_max > 950.0:
        break

t_total_comp = time.perf_counter() - t_start
print(f" Concluído em {t_total_comp*1000:.1f} ms.")

T_final_max = float(np.max(T0))
T_final_C = T_final_max - 273.15
ni_final = float(ni(T_final_max))
mu_n_final = float(mu_n(T_final_max))
sigma_final = float(conductivity(T_final_max))
Eg_final = float(Eg(T_final_max))
_, _, _, _, EF_final = fermi_levels(T_final_max)
Dn_final, Dp_final = einstein_diffusion(T_final_max)
tau_e_final, _ = relaxation_times(T_final_max)
Ln_final, _ = diffusion_lengths(T_final_max)

print(f"  Tempo Físico até Colapso: {elapsed_s*1e3:.2f} ms ({len(hist_t_ms)*N_BATCH} passos FDM)")
print(f"  Temperatura Máxima Atingida: {T_final_C:.1f} °C (TjMax superado)")

# =============================================================================
# 6. GERAÇÃO VISUAL PADRÃO ABNT / IEEE (Preto e Branco / Fundo Branco / Alta Legibilidade)
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 9.0,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#000000",
    "axes.linewidth": 0.8,
    "grid.color": "#D5D5D5",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
})

fig = plt.figure(figsize=(15.5, 9.8), dpi=300)

# Título Superior Obrigatório ABNT (NBR 14724)
fig.text(
    0.5, 0.978,
    "Figura 1 – Simulação térmica bidimensional e resposta de transporte semicondutor no die do processador",
    ha="center", va="top", fontsize=11.5, fontweight="bold", color="#000000"
)

gs = gridspec.GridSpec(
    2, 3, figure=fig,
    left=0.06, right=0.96, top=0.935, bottom=0.085,
    hspace=0.36, wspace=0.28, height_ratios=[1.7, 1.0]
)

# --- PAINEL 1: MAPA TÉRMICO 2D COM ISOTERMAS E BLOCOS FUNCIONAIS ---
ax_map = fig.add_subplot(gs[0, :2])
extent = [0, NX * DX * 1e3, 0, NY * DY * 1e3]
T_grid_C = (T0 - 273.15).T

im = ax_map.imshow(
    T_grid_C, origin="lower", extent=extent, cmap="inferno",
    vmin=80.0, vmax=max(T_final_C * 1.02, 560.0), aspect="auto", interpolation="bicubic"
)

# Contornos Isotérmicos
cs = ax_map.contour(
    np.linspace(0, NX * DX * 1e3, NX), np.linspace(0, NY * DY * 1e3, NY),
    T_grid_C, levels=[150, 250, 350, 450, 520],
    colors=["#FFFFFF", "#FFFFFF", "#FFFFFF", "#000000", "#000000"],
    linewidths=0.9, linestyles="--"
)
ax_map.clabel(cs, inline=True, fontsize=7.5, fmt="%d °C")

# Rótulos dos Blocos em caixas brancas com bordas pretas finas
for name, dom, x0, x1, y0, y1 in BLOCKS:
    x0_mm, x1_mm = x0 * DX * 1e3, x1 * DX * 1e3
    y0_mm, y1_mm = y0 * DY * 1e3, y1 * DY * 1e3
    w_mm, h_mm = x1_mm - x0_mm, y1_mm - y0_mm
    
    rect = mpatches.Rectangle((x0_mm, y0_mm), w_mm, h_mm, fill=False, edgecolor="#FFFFFF", lw=0.75, alpha=0.95)
    ax_map.add_patch(rect)
    
    if w_mm >= 2.0 and h_mm >= 0.8:
        cx = (x0_mm + x1_mm) / 2.0
        cy = (y0_mm + y1_mm) / 2.0
        fsize = 5.8 if h_mm < 1.5 else 6.8
        
        # Desloca rótulo do P-Core 0 para prevenir sobreposição com o hotspot
        if name == "P-Core 0":
            cy = y0_mm + 0.32 * h_mm
            
        ax_map.text(
            cx, cy, name, ha="center", va="center", fontsize=fsize, fontweight="bold", color="#000000",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="#FFFFFF", edgecolor="#000000", lw=0.55, alpha=0.92)
        )

# Marcador do Hotspot Térmico Máximo
hot_idx = np.unravel_index(np.argmax(T0), T0.shape)
hx, hy = hot_idx[0] * DX * 1e3, hot_idx[1] * DY * 1e3
ax_map.plot(hx, hy, marker="o", markersize=8, markerfacecolor="none", markeredgecolor="#FFFFFF", markeredgewidth=1.5)
ax_map.plot(hx, hy, marker="x", markersize=7, color="#FFFFFF", markeredgewidth=1.3)
ax_map.annotate(
    f"Hotspot: {T_final_C:.1f} °C", xy=(hx, hy), xytext=(hx + 2.0, hy + 0.7),
    fontsize=8, fontweight="bold", color="#000000",
    bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="#000000", lw=0.75),
    arrowprops=dict(arrowstyle="->", color="#FFFFFF", lw=1.1)
)

ax_map.set_xlabel("Comprimento do Die $x$ (mm)", fontsize=9.5)
ax_map.set_ylabel("Largura do Die $y$ (mm)", fontsize=9.5)
ax_map.set_title(f"a) Campo térmico bidimensional transiente no colapso ($t = {elapsed_s*1e3:.1f}\\text{{ ms}}$)", fontsize=10, loc="left", pad=6)

cbar = fig.colorbar(im, ax=ax_map, fraction=0.030, pad=0.015)
cbar.set_label("Temperatura $T$ (°C)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# --- PAINEL 2: QUADRO TÉCNICO ABNT (TABELA DE FÍSICA E POTÊNCIA) ---
ax_table = fig.add_subplot(gs[0, 2])
ax_table.axis("off")

ax_table.text(
    0.5, 1.0, "Quadro 1 – Parâmetros Físicos e Balanço Térmico no Colapso",
    ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#000000"
)

# Três regras horizontais segundo normas do IBGE / ABNT
ax_table.axhline(0.97, color="#000000", lw=1.3)
ax_table.axhline(0.89, color="#000000", lw=0.8)

# Cabeçalhos
ax_table.text(0.02, 0.93, "Parâmetro / Propriedade", va="center", fontsize=8.0, fontweight="bold")
ax_table.text(0.58, 0.93, "Símbolo", va="center", fontsize=8.0, fontweight="bold")
ax_table.text(0.98, 0.93, "Valor no Colapso", va="center", ha="right", fontsize=8.0, fontweight="bold")

table_data = [
    ("Temperatura Máxima", "$T_{\\mathrm{máx}}$", f"{T_final_max:.1f} K ({T_final_C:.1f} °C)"),
    ("Bandgap do Silício", "$E_g(T)$", f"{Eg_final:.4f} eV"),
    ("Conc. Intrínseca", "$n_i(T)$", "$1{,}00 \\times 10^{17}\\text{ cm}^{-3}$"),
    ("Dopagem de Doadores", "$N_d$", "$1{,}00 \\times 10^{17}\\text{ cm}^{-3}$"),
    ("Razão de Portadores", "$n_i / N_d$", "$\\mathbf{1{,}00}$ (Limiar Crítico)"),
    ("Mobilidade de Elétrons", "$\\mu_n(T)$", f"{mu_n_final:.1f} cm$^2$/(V·s)"),
    ("Condutividade Elétrica", "$\\sigma(T)$", f"{sigma_final:.4f} S/cm"),
    ("Nível de Fermi", "$E_F - E_i$", f"{EF_final:.4f} eV"),
    ("Coef. de Difusão", "$D_n$", f"{Dn_final:.2f} cm$^2$/s"),
    ("Tempo até Colapso", "$t_{\\mathrm{falha}}$", f"{elapsed_s*1e3:.1f} ms"),
]

y_row = 0.84
for param, sym, val in table_data:
    is_crit = "Razão" in param or "Temperatura" in param
    fw = "bold" if is_crit else "normal"
    ax_table.text(0.02, y_row, param, va="center", fontsize=7.6, fontweight=fw)
    ax_table.text(0.58, y_row, sym, va="center", fontsize=7.6)
    ax_table.text(0.98, y_row, val, va="center", ha="right", fontsize=7.6, fontweight=fw)
    y_row -= 0.044

ax_table.axhline(y_row + 0.02, color="#000000", lw=0.8)
y_row -= 0.01

# Sub-seção de potência
ax_table.text(0.02, y_row, "Dissipação por Bloco (Carga Gaming = 125 W):", va="center", fontsize=7.8, fontweight="bold")
y_row -= 0.040

power_summary = [
    ("• P-Cores (8× Raptor Cove)", "82 W", "65,6% (Hotspot)"),
    ("• L3 Smart Cache & Ring Bus", "18 W", "14,4%"),
    ("• E-Cores (8× Gracemont)", "14 W", "11,2%"),
    ("• IMC, PCIe & SysAgent", "9 W", "7,2%"),
    ("• iGPU (UHD 770 ociosa)", "2 W", "1,6%"),
]

for blk_name, pw, pct in power_summary:
    ax_table.text(0.04, y_row, blk_name, va="center", fontsize=7.3)
    ax_table.text(0.72, y_row, pw, va="center", fontsize=7.3)
    ax_table.text(0.98, y_row, pct, va="center", ha="right", fontsize=7.3)
    y_row -= 0.038

ax_table.axhline(y_row + 0.02, color="#000000", lw=1.3)

# Nota técnica de rodapé do quadro
ax_table.text(
    0.02, y_row - 0.02,
    "Nota: Transição ao regime intrínseco ($n_i \\geq N_d$) anula a barreira de corte dos transistores.",
    va="top", fontsize=6.8, style="italic"
)

# --- PAINEL 3: EVOLUÇÃO TEMPORAL DE TEMPERATURA ---
ax_temp = fig.add_subplot(gs[1, 0])
ax_temp.plot(hist_t_ms, hist_Tmax_C, color="#000000", lw=1.8, label="$T_{\\mathrm{máx}}(t)$")
ax_temp.axhline(100.0, color="#555555", ls="--", lw=1.0, label="TjMax (100 °C)")
ax_temp.axhline(T_fail_C, color="#000000", ls=":", lw=1.2, label=f"Falha Física ({T_fail_C:.1f} °C)")
ax_temp.plot(hist_t_ms[-1], hist_Tmax_C[-1], marker="s", markersize=5, color="#000000")
ax_temp.set_xlabel("Tempo transiente $t$ (ms)", fontsize=9)
ax_temp.set_ylabel("Temperatura máxima (°C)", fontsize=9)
ax_temp.set_title("b) Aquecimento adiabático transitório", fontsize=9.5, loc="left", pad=5)
ax_temp.grid(True)
ax_temp.legend(loc="upper left", fontsize=7.5, frameon=True, edgecolor="#000000", facecolor="#FFFFFF")

# --- PAINEL 4: RAZÃO n_i / N_d ---
ax_ni = fig.add_subplot(gs[1, 1])
ax_ni.semilogy(hist_t_ms, hist_ni_ratio, color="#000000", lw=1.8, label="$n_i(T) / N_d$")
ax_ni.axhline(1.0, color="#555555", ls="--", lw=1.1, label="$n_i = N_d$ (Colapso)")
ax_ni.axhline(0.01, color="#888888", ls=":", lw=0.9, label="Limiar 1%")
ax_ni.plot(hist_t_ms[-1], hist_ni_ratio[-1], marker="s", markersize=5, color="#000000")
ax_ni.set_xlabel("Tempo transiente $t$ (ms)", fontsize=9)
ax_ni.set_ylabel("Razão $n_i / N_d$ (adimensional)", fontsize=9)
ax_ni.set_title("c) Transição extrínseco-intrínseco", fontsize=9.5, loc="left", pad=5)
ax_ni.grid(True)
ax_ni.legend(loc="lower right", fontsize=7.5, frameon=True, edgecolor="#000000", facecolor="#FFFFFF")

# --- PAINEL 5: MOBILIDADE E CONDUTIVIDADE ---
ax_mob = fig.add_subplot(gs[1, 2])
l1, = ax_mob.plot(hist_t_ms, hist_mu_n, color="#000000", lw=1.8, label="$\\mu_n$")
ax_mob.set_xlabel("Tempo transiente $t$ (ms)", fontsize=9)
ax_mob.set_ylabel("Mobilidade $\\mu_n$ [cm$^2$/(V·s)]", color="#000000", fontsize=9)
ax_mob.grid(True)

ax_sig = ax_mob.twinx()
l2, = ax_sig.plot(hist_t_ms, hist_sigma, color="#444444", lw=1.6, ls="--", label="$\\sigma$")
ax_sig.set_ylabel("Condutividade $\\sigma$ [S/cm]", color="#444444", fontsize=9)
ax_mob.set_title("d) Degradação de transporte", fontsize=9.5, loc="left", pad=5)
ax_mob.legend([l1, l2], ["$\\mu_n$ (esq.)", "$\\sigma$ (dir.)"], loc="center right", fontsize=7.5, frameon=True, edgecolor="#000000", facecolor="#FFFFFF")

# Rodapé Padrão ABNT (Fonte e Autoria)
fig.text(
    0.06, 0.015,
    "Fonte: Dados obtidos via simulação numérica FDM e modelos de transporte semicondutor, elaborada pelos autores (2026).",
    ha="left", va="bottom", fontsize=8.5, style="italic", color="#000000"
)

out_file = "simulacao_termica_resultado.png"
plt.savefig(out_file, dpi=300, facecolor="#FFFFFF")
plt.close(fig)

print(f"\n  ✓ Imagem gerada com sucesso em padrão ABNT: {out_file}")
print("=" * 72)
