# -*- coding: utf-8 -*-
"""
==============================================================================
 SIMULAÇÃO TÉRMICA 2D — Intel Core i7-13700K (Raptor Lake-S)
 Disciplina: Materiais e Dispositivos Elétricos — Engenharia Física
==============================================================================
"""

import sys
import io
import time as _time

# Força UTF-8 no console do Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend não-bloqueante para salvar a imagem sem travar o processo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from numba import njit, prange
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONSTANTES FÍSICAS E SEMICONDUTORAS (Silício)
# =============================================================================
Q_E       = 1.602176634e-19   # Carga elementar [C]
K_B_EV    = 8.617333262e-5    # Constante de Boltzmann [eV/K]
K_B_J     = 1.380649e-23      # Constante de Boltzmann [J/K]

# --- Silício: propriedades térmicas (300 K) ---
RHO_SI    = 2329.0            # Densidade [kg/m³]
CP_SI_300 = 700.0             # Calor específico a 300K [J/(kg·K)]
K_SI_300  = 148.0             # Condutividade térmica a 300K [W/(m·K)]

# --- Silício: propriedades semicondutoras ---
EG0_SI    = 1.1695            # Bandgap T=0K [eV] — Varshni
ALPHA_V   = 4.73e-4           # Varshni α [eV/K]
BETA_V    = 636.0             # Varshni β [K]
NC_300    = 2.8e19            # Densidade efetiva estados BC a 300K [cm⁻³]
NV_300    = 1.04e19           # Densidade efetiva estados BV a 300K [cm⁻³]
MU_N_300  = 1417.0            # Mobilidade elétrons a 300K [cm²/(V·s)] — Jacoboni-Canali
MU_P_300  = 470.5             # Mobilidade buracos a 300K [cm²/(V·s)]
N_D       = 1e17              # Concentração de doadores (tipo-n) [cm⁻³]
NI_300    = 1.5e10            # Concentração intrínseca a 300K [cm⁻³]

# =============================================================================
# 2. MODELOS FÍSICOS DA DISCIPLINA
# =============================================================================
def Eg(T):
    """Bandgap Varshni: E_g(T) = E_g(0) - α·T² / (T + β) [eV]"""
    T = np.asarray(T, dtype=float)
    return EG0_SI - ALPHA_V * T**2 / (T + BETA_V)

def ni(T):
    """Concentração intrínseca: n_i(T) = √(Nc·Nv) · exp(-Eg / 2kBT) [cm⁻³]"""
    T = np.asarray(T, dtype=float)
    T_safe = np.maximum(T, 1.0)
    Nc = NC_300 * (T_safe / 300.0)**1.5
    Nv = NV_300 * (T_safe / 300.0)**1.5
    return np.sqrt(Nc * Nv) * np.exp(-Eg(T_safe) / (2.0 * K_B_EV * T_safe))

def mu_n(T):
    """Mobilidade elétrons Jacoboni-Canali: μ_n(T) = μ_n(300) · (T/300)^(-2.42)"""
    T = np.asarray(T, dtype=float)
    return MU_N_300 * (T / 300.0)**(-2.42)

def mu_p(T):
    """Mobilidade buracos: μ_p(T) = μ_p(300) · (T/300)^(-2.20)"""
    T = np.asarray(T, dtype=float)
    return MU_P_300 * (T / 300.0)**(-2.20)

def carrier_concentrations(T, Nd=N_D):
    """Neutralidade de carga (Si tipo-n): n_o = Nd/2 + √((Nd/2)² + ni²)"""
    T = np.asarray(T, dtype=float)
    ni_T = ni(T)
    n_o = 0.5 * Nd + np.sqrt((0.5 * Nd)**2 + ni_T**2)
    p_o = ni_T**2 / n_o
    return n_o, p_o

def conductivity(T, Nd=N_D):
    """Condutividade: σ(T) = e · (n_o·μ_n + p_o·μ_p) [S/cm]"""
    n_o, p_o = carrier_concentrations(T, Nd)
    return Q_E * (n_o * mu_n(T) + p_o * mu_p(T))

def fermi_level_above_intrinsic(T, Nd=N_D):
    """E_F - E_i = kB·T · ln(n_o / n_i) [eV]"""
    T = np.asarray(T, dtype=float)
    T_safe = np.maximum(T, 1.0)
    ni_T = ni(T_safe)
    n_o, _ = carrier_concentrations(T_safe, Nd)
    return K_B_EV * T_safe * np.log(np.maximum(n_o / ni_T, 1e-300))

# =============================================================================
# 3. LAYOUT E POTÊNCIAS DO i7-13700K (Gaming Load ~125 W sem arrefecimento)
# =============================================================================
NX, NY  = 120, 60             # Dimensões da grade
DX = DY = 200e-6              # 200 µm por célula -> Die de 24 mm x 12 mm
DZ = 500e-6                   # Espessura do substrato: 500 µm
T_AMB   = 300.0               # Temperatura inicial (26.85 °C)
T_THROTTLE_K = 373.15         # TjMax Intel = 100 °C

# Potências por bloco funcional (Fontes: Intel Arch Day 2022, AnandTech, Chips&Cheese)
GAMING_POWER = {
    "P-Core":    82.0,   # 8 P-Cores Raptor Cove (~10.25 W/core) -> Principal gerador de calor (~66%)
    "L3-Cache":  18.0,   # 30 MB L3 Smart Cache + Ring Bus a 4.5+ GHz (~14%)
    "E-Cluster": 14.0,   # 2 clusters Gracemont (8 E-cores, ~1.75 W/core) (~11%)
    "IMC-IO":     9.0,   # Memory Controller DDR4/5 + PCIe Gen5 + System Agent (~7%)
    "iGPU":       2.0,   # UHD 770 ociosa durante gaming com dGPU (~2%)
}
TOTAL_POWER = sum(GAMING_POWER.values())  # 125 W

# Coordenadas dos blocos no Die (em células da grade)
BLOCKS = [
    # 8 P-Cores (Raptor Cove) — Superior Esquerdo (alta densidade de potência)
    {"name": "P-Core 0", "x0":  2, "x1": 16, "y0": 42, "y1": 57, "domain": "P-Core"},
    {"name": "P-Core 1", "x0": 17, "x1": 31, "y0": 42, "y1": 57, "domain": "P-Core"},
    {"name": "P-Core 2", "x0": 32, "x1": 46, "y0": 42, "y1": 57, "domain": "P-Core"},
    {"name": "P-Core 3", "x0": 47, "x1": 61, "y0": 42, "y1": 57, "domain": "P-Core"},
    {"name": "P-Core 4", "x0":  2, "x1": 16, "y0": 26, "y1": 41, "domain": "P-Core"},
    {"name": "P-Core 5", "x0": 17, "x1": 31, "y0": 26, "y1": 41, "domain": "P-Core"},
    {"name": "P-Core 6", "x0": 32, "x1": 46, "y0": 26, "y1": 41, "domain": "P-Core"},
    {"name": "P-Core 7", "x0": 47, "x1": 61, "y0": 26, "y1": 41, "domain": "P-Core"},

    # 2 Clusters de E-Cores (Gracemont) — Superior Direito
    {"name": "E-Cluster 0\n(4 cores)", "x0": 62, "x1": 85,  "y0": 35, "y1": 57, "domain": "E-Cluster"},
    {"name": "E-Cluster 1\n(4 cores)", "x0": 86, "x1": 109, "y0": 35, "y1": 57, "domain": "E-Cluster"},

    # Ring Bus — Interconector horizontal central
    {"name": "Ring Bus Interconnect", "x0":  2, "x1": 109, "y0": 20, "y1": 25, "domain": "L3-Cache"},

    # L3 Cache (30 MB Smart Cache compartilhado)
    {"name": "L3 Smart Cache (30 MB)", "x0":  2, "x1": 109, "y0":  8, "y1": 19, "domain": "L3-Cache"},

    # I/O, Memória e System Agent — Faixa Inferior
    {"name": "Mem Ctrl (DDR5)", "x0":  2, "x1": 28,  "y0":  2, "y1":  7, "domain": "IMC-IO"},
    {"name": "PCIe 5.0 Controller", "x0": 29, "x1": 60,  "y0":  2, "y1":  7, "domain": "IMC-IO"},
    {"name": "System Agent (PCU)",  "x0": 61, "x1": 85,  "y0":  2, "y1":  7, "domain": "IMC-IO"},

    # iGPU (UHD Graphics 770) — Inferior Direito
    {"name": "iGPU (UHD 770)",      "x0": 86, "x1": 109, "y0":  2, "y1": 34, "domain": "iGPU"},
]

DOMAIN_COLORS = {
    "P-Core":    "#D32F2F",   # Vermelho (máximo calor)
    "L3-Cache":  "#F57C00",   # Laranja (moderado-alto)
    "E-Cluster": "#1976D2",   # Azul (eficiência)
    "IMC-IO":    "#388E3C",   # Verde (I/O)
    "iGPU":      "#7B1FA2",   # Roxo (idle/pouco calor)
}

DOMAIN_LABELS = {
    "P-Core":    "P-Cores (8× Raptor Cove) — 82 W (66%)",
    "L3-Cache":  "L3 Cache + Ring Bus — 18 W (14%)",
    "E-Cluster": "E-Cores (8× Gracemont) — 14 W (11%)",
    "IMC-IO":    "IMC + PCIe + System Agent — 9 W (7%)",
    "iGPU":      "iGPU (UHD 770, ociosa) — 2 W (2%)",
}

def build_power_map():
    P_dyn = np.zeros((NX, NY), dtype=float)
    cell_vol = DX * DY * DZ
    domain_cells = {d: 0 for d in GAMING_POWER}
    domain_masks = {d: np.zeros((NX, NY), dtype=bool) for d in GAMING_POWER}

    for blk in BLOCKS:
        d = blk["domain"]
        mask = np.zeros((NX, NY), dtype=bool)
        mask[blk["x0"]:blk["x1"], blk["y0"]:blk["y1"]] = True
        domain_masks[d] |= mask
        domain_cells[d] += (blk["x1"] - blk["x0"]) * (blk["y1"] - blk["y0"])

    for d, total_power in GAMING_POWER.items():
        n_cells = max(domain_cells[d], 1)
        power_per_cell = total_power / n_cells
        P_dyn[domain_masks[d]] += power_per_cell

    return P_dyn / cell_vol, domain_masks

# =============================================================================
# 4. SOLVER FDM 2D PARALELO COM NUMBA (C-Compiled via LLVM)
# =============================================================================
_alpha_300 = K_SI_300 / (RHO_SI * CP_SI_300)
DT_FIXED   = DX**2 / (4.0 * _alpha_300) * 0.40  # Estabilidade de Von Neumann
_K_CONST   = K_SI_300

@njit(parallel=True)
def _fdm_kernel(T, P_gen_vol, dt, k_const, rho, cp, dx):
    """
    Equação da Difusão Térmica 2D com Geração:
    ρ·cp · ∂T/∂t = k·∇²T + Q_gen
    Fronteiras de Neumann (fluxo zero): ausência de arrefecimento.
    """
    NX_loc, NY_loc = T.shape
    T_new = T.copy()
    inv_dx2 = 1.0 / (dx * dx)
    ck = dt * k_const / (rho * cp) * inv_dx2
    cq = dt / (rho * cp)

    for i in prange(1, NX_loc - 1):
        for j in range(1, NY_loc - 1):
            lap = (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - 4.0 * T[i, j])
            T_new[i, j] = T[i, j] + ck * lap + cq * P_gen_vol[i, j]

    # Condição adiabática (Neumann: dT/dn = 0) nas 4 bordas
    for j in range(NY_loc):
        T_new[0, j]        = T_new[1, j]
        T_new[NX_loc-1, j] = T_new[NX_loc-2, j]
    for i in range(NX_loc):
        T_new[i, 0]        = T_new[i, 1]
        T_new[i, NY_loc-1] = T_new[i, NY_loc-2]

    return T_new

def warmup_numba(T, P_gen_vol):
    """Compila o kernel na primeira execução."""
    _ = _fdm_kernel(T, P_gen_vol, DT_FIXED, _K_CONST, RHO_SI, CP_SI_300, DX)

# =============================================================================
# 5. EXECUÇÃO DA SIMULAÇÃO
# =============================================================================
print("=" * 70)
print("  SIMULAÇÃO TÉRMICA 2D — Intel Core i7-13700K")
print("  Disciplina: Materiais e Dispositivos Elétricos (Engenharia Física)")
print("=" * 70)

T_fail_K = brentq(lambda T: float(ni(T)) - N_D, 400, 1500)
T_fail_C = T_fail_K - 273.15

print(f"  Ponto de Colapso Físico (n_i = N_d): {T_fail_K:.1f} K ({T_fail_C:.1f} °C)")
print(f"  Potência Dissipada em Carga Gaming: {TOTAL_POWER:.0f} W (sem arrefecimento)")

T_field = np.full((NX, NY), T_AMB, dtype=float)
P_gen_vol, domain_masks = build_power_map()

# JIT warmup
print("  Compilando kernel FDM em código C nativo via Numba...")
t_w = _time.perf_counter()
warmup_numba(T_field, P_gen_vol)
print(f"  Kernel compilado em {_time.perf_counter() - t_w:.2f} s")

# Listas de histórico temporal
hist_times    = []
hist_Tmax_C   = []
hist_ni_ratio = []
hist_mu_n_T   = []
hist_sigma_T  = []

elapsed_us = 0.0
failed = False
failure_reason = ""
N_BATCH = 100
MAX_LOTES = 10000

t_start = _time.perf_counter()
for lote in range(MAX_LOTES):
    # Executa lote de passos paralelos
    for _ in range(N_BATCH):
        T_field = _fdm_kernel(T_field, P_gen_vol, DT_FIXED, _K_CONST, RHO_SI, CP_SI_300, DX)
        elapsed_us += DT_FIXED * 1e6

    T_max = float(np.max(T_field))
    ni_T  = float(ni(T_max))
    r_ni  = ni_T / N_D

    hist_times.append(elapsed_us)
    hist_Tmax_C.append(T_max - 273.15)
    hist_ni_ratio.append(max(r_ni, 1e-15))
    hist_mu_n_T.append(float(mu_n(T_max)))
    hist_sigma_T.append(float(conductivity(T_max)))

    # Critério de parada: transição para regime intrínseco (n_i >= N_d)
    if ni_T >= N_D:
        failed = True
        failure_reason = "Regime Intrínseco Atingido (n_i ≥ N_d)"
        break
    if T_max > 950.0:
        failed = True
        failure_reason = "Limite de Integridade Térmica"
        break

t_total = _time.perf_counter() - t_start
T_max_final   = float(np.max(T_field))
T_max_final_C = T_max_final - 273.15
ni_final      = float(ni(T_max_final))
mu_n_final    = float(mu_n(T_max_final))
mu_p_final    = float(mu_p(T_max_final))
n_o_final, p_o_final = carrier_concentrations(T_max_final)
sigma_final   = float(conductivity(T_max_final))
EF_final      = float(fermi_level_above_intrinsic(T_max_final))
Eg_final      = float(Eg(T_max_final))
Dn_final      = mu_n_final * K_B_EV * T_max_final
ni_ratio_final = ni_final / N_D
sigma_300 = float(conductivity(300.0))

print(f"  Simulação calculada com sucesso em {t_total:.2f} s ({elapsed_us/1e6:.3f} s físicos).")
print(f"  T_max final atingida: {T_max_final_C:.1f} °C (Critério: {failure_reason})")

# =============================================================================
# 6. GERAÇÃO DO PAINEL GRÁFICO (Tema Claro e Visual Limpo)
# =============================================================================
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "axes.facecolor":   "#FFFFFF",
    "figure.facecolor": "#F8F9FA",
    "axes.edgecolor":   "#B0BEC5",
    "axes.labelcolor":  "#263238",
    "xtick.color":      "#37474F",
    "ytick.color":      "#37474F",
    "text.color":       "#263238",
    "grid.color":       "#ECEFF1",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
})

fig = plt.figure(figsize=(19, 11), facecolor="#F8F9FA")

# Título limpo e direto, sem sobreposição
fig.suptitle(
    "Simulação Térmica 2D — Intel Core i7-13700K (Cenário Gaming sem Arrefecimento)",
    fontsize=15, fontweight="bold", color="#0D47A1", y=0.975
)

# Grid layout: 2 linhas, 3 colunas
# Top: Heatmap (2 colunas) + Painel de Física & Balanço de Potência (1 coluna)
# Bottom: 3 gráficos temporais da física do semicondutor
gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    left=0.05, right=0.98,
    top=0.92, bottom=0.08,
    hspace=0.42, wspace=0.30,
    height_ratios=[2.2, 1.0],
)

# -----------------------------------------------------------------------------
# PAINEL 1: MAPA TÉRMICO 2D COM ISOTERMAS E BLOCOS DO DIE
# -----------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[0, :2])
extent = [0, NX * DX * 1e3, 0, NY * DY * 1e3]  # mm

# Colormap térmico FLIR/Turbo (perceptualmente intuitivo: azul=frio, amarelo=médio, vermelho/vinho=máximo)
T_data_C = (T_field - 273.15).T
im = ax_map.imshow(
    T_data_C, origin="lower", extent=extent,
    cmap="turbo", vmin=100.0, vmax=max(T_max_final_C * 1.02, 550.0),
    aspect="auto", interpolation="bicubic", alpha=0.88
)

cbar = fig.colorbar(im, ax=ax_map, fraction=0.035, pad=0.02)
cbar.set_label("Temperatura (°C)", fontsize=10, fontweight="bold", color="#263238")
cbar.ax.tick_params(labelsize=8.5)

# Linhas de isotermas para visualizar claramente a propagação do calor
contour_levels = [150, 250, 350, 450, 520]
cs = ax_map.contour(
    np.linspace(0, NX * DX * 1e3, NX),
    np.linspace(0, NY * DY * 1e3, NY),
    T_data_C,
    levels=[lvl for lvl in contour_levels if lvl < T_max_final_C],
    colors=["#FFFFFF", "#FFFFFF", "#212121", "#FFFFFF", "#FFFFFF"],
    linewidths=1.1, linestyles="--", alpha=0.85
)
ax_map.clabel(cs, inline=True, fontsize=8, fmt="%d°C", colors="#212121")

# Bordas finas dos blocos funcionais com etiquetas de alto contraste
for blk in BLOCKS:
    col = DOMAIN_COLORS[blk["domain"]]
    x0_mm = blk["x0"] * DX * 1e3
    x1_mm = blk["x1"] * DX * 1e3
    y0_mm = blk["y0"] * DY * 1e3
    y1_mm = blk["y1"] * DY * 1e3
    w_mm = x1_mm - x0_mm
    h_mm = y1_mm - y0_mm

    # Retângulo de contorno do bloco
    rect = mpatches.Rectangle(
        (x0_mm, y0_mm), w_mm, h_mm,
        linewidth=1.4, edgecolor=col, facecolor="none", zorder=3
    )
    ax_map.add_patch(rect)

    # Etiqueta central com badge branco semi-transparente para legibilidade perfeita
    cx = (x0_mm + x1_mm) / 2.0
    cy = (y0_mm + y1_mm) / 2.0
    name = blk["name"].replace("\n", " ")

    if w_mm > 2.5 and h_mm > 1.2:
        ax_map.text(
            cx, cy, name, ha="center", va="center",
            fontsize=7, fontweight="bold", color="#1A237E",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.90, lw=1.0),
            zorder=4
        )

# Ponto mais quente (Hotspot)
hot_idx = np.unravel_index(np.argmax(T_field), T_field.shape)
hot_x = hot_idx[0] * DX * 1e3
hot_y = hot_idx[1] * DY * 1e3
ax_map.plot(hot_x, hot_y, marker="*", color="#D50000", markersize=14, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax_map.annotate(
    f"Hotspot Máx: {T_max_final_C:.1f} °C",
    xy=(hot_x, hot_y), xytext=(hot_x + 1.2, hot_y - 1.2),
    fontsize=8.5, fontweight="bold", color="#B71C1C",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFEBEE", edgecolor="#B71C1C", lw=1.2),
    arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.2),
    zorder=6
)

ax_map.set_xlabel("Largura do Die X (mm)", fontsize=10, fontweight="bold")
ax_map.set_ylabel("Altura do Die Y (mm)", fontsize=10, fontweight="bold")
ax_map.set_title(
    f"Distribuição Térmica 2D do Die — Fim da Simulação (T_máx = {T_max_final_C:.1f} °C)",
    fontsize=12, fontweight="bold", color="#0D47A1", pad=8
)
ax_map.set_xlim(0, NX * DX * 1e3)
ax_map.set_ylim(0, NY * DY * 1e3)

# -----------------------------------------------------------------------------
# PAINEL 2: FÍSICA DO SEMICONDUTOR & ARQUITETURA DE POTÊNCIA
# -----------------------------------------------------------------------------
ax_phys = fig.add_subplot(gs[0, 2])
ax_phys.set_facecolor("#FFFFFF")
ax_phys.axis("off")

# Borda elegante para o painel de dados
panel_rect = mpatches.FancyBboxPatch(
    (0.01, 0.01), 0.98, 0.98,
    boxstyle="round,pad=0.02",
    linewidth=1.5, edgecolor="#90CAF9", facecolor="#FAFCFF",
    transform=ax_phys.transAxes, zorder=0
)
ax_phys.add_patch(panel_rect)

# Cabeçalho da física
ax_phys.text(
    0.5, 0.95, "Física do Silício & Colapso Lógico",
    ha="center", va="top", fontsize=11, fontweight="bold", color="#0D47A1",
    transform=ax_phys.transAxes
)

mu_drop = (1.0 - mu_n_final / MU_N_300) * 100.0

physics_rows = [
    ("Temperatura Máxima (T_máx):", f"{T_max_final:.1f} K  ({T_max_final_C:.1f} °C)"),
    ("Bandgap E_g(T) [Varshni]:", f"{Eg_final:.4f} eV  (era 1.12 eV)"),
    ("Conc. Intrínseca n_i(T):", f"{ni_final:.2e} cm^-3"),
    ("Dopagem de Doadores N_d:", f"{N_D:.1e} cm^-3"),
    ("Razão Crítica n_i / N_d:", f"{ni_ratio_final:.2f}  (≥ 1.0 → FALHA)"),
    ("Conc. Eletrônica Majoritária n_o:", f"{n_o_final:.2e} cm^-3"),
    ("Conc. Lacunas Minoritária p_o:", f"{p_o_final:.2e} cm^-3"),
    ("Mobilidade de Elétrons μ_n(T):", f"{mu_n_final:.0f} cm²/(V·s)  (-{mu_drop:.1f}%)"),
    ("Condutividade Elétrica σ(T):", f"{sigma_final:.4f} S/cm"),
    ("Nível de Fermi (E_F - E_i):", f"{EF_final:.4f} eV  (→ 0 no colapso)"),
    ("Coef. Difusão D_n (Einstein):", f"{Dn_final:.2f} cm²/s"),
]

y_pos = 0.88
for label, val in physics_rows:
    is_crit = "Razão Crítica" in label or "Temperatura Máxima" in label
    col_t = "#B71C1C" if is_crit else "#37474F"
    font_w = "bold" if is_crit else "normal"
    ax_phys.text(0.06, y_pos, label, fontsize=8.2, color="#455A64", va="top", transform=ax_phys.transAxes)
    ax_phys.text(0.94, y_pos, val, fontsize=8.2, color=col_t, fontweight=font_w, va="top", ha="right", transform=ax_phys.transAxes)
    y_pos -= 0.046

# Divisor
ax_phys.plot([0.06, 0.94], [y_pos, y_pos], color="#CFD8DC", lw=1.0, transform=ax_phys.transAxes)
y_pos -= 0.025

# Seção de Dissipação Térmica dos Blocos (Responde ao tema da apresentação)
ax_phys.text(
    0.5, y_pos, "Dissipação por Bloco (Carga Gaming = 125 W)",
    ha="center", va="top", fontsize=9.5, fontweight="bold", color="#1565C0",
    transform=ax_phys.transAxes
)
y_pos -= 0.045

power_rows = [
    ("P-Cores (8× Raptor Cove):", "82 W (66%)  — Hotspot central", DOMAIN_COLORS["P-Core"]),
    ("L3 Cache + Ring Bus:", "18 W (14%)  — Ponte térmica", DOMAIN_COLORS["L3-Cache"]),
    ("E-Cores (8× Gracemont):", "14 W (11%)  — Eficiência", DOMAIN_COLORS["E-Cluster"]),
    ("IMC + PCIe + System Agent:", "9 W (7%)   — Periferia I/O", DOMAIN_COLORS["IMC-IO"]),
    ("iGPU (UHD 770 ociosa):", "2 W (2%)   — Power-gated", DOMAIN_COLORS["iGPU"]),
]

for name_b, info_b, cor_b in power_rows:
    ax_phys.plot(0.07, y_pos - 0.015, marker="s", markersize=6, color=cor_b, transform=ax_phys.transAxes)
    ax_phys.text(0.12, y_pos, name_b, fontsize=7.8, color="#263238", va="top", transform=ax_phys.transAxes)
    ax_phys.text(0.94, y_pos, info_b, fontsize=7.8, color="#263238", fontweight="bold", va="top", ha="right", transform=ax_phys.transAxes)
    y_pos -= 0.040

# Badge de Colapso Lógico
ax_phys.text(
    0.5, 0.05,
    f"STATUS: REGIME INTRÍNSECO ATINGIDO (FALHA LÓGICA)\n"
    f"Geração térmica supera dopagem (n_i ≥ N_d) em {elapsed_us/1e3:.1f} ms",
    ha="center", va="center", fontsize=8.2, fontweight="bold", color="#B71C1C",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#D32F2F", lw=1.5),
    transform=ax_phys.transAxes
)

# -----------------------------------------------------------------------------
# PAINEL 3: TEMPERATURA MÁXIMA VS TEMPO
# -----------------------------------------------------------------------------
ax_temp = fig.add_subplot(gs[1, 0])
t_ms = np.array(hist_times) / 1e3  # converte para milissegundos para facilitar leitura

ax_temp.plot(t_ms, hist_Tmax_C, color="#D32F2F", lw=2.2, label="T_máx (°C)")
ax_temp.axhline(100.0, color="#FFA000", ls="--", lw=1.4, label="TjMax Intel (100 °C)")
ax_temp.axhline(T_fail_C, color="#B71C1C", ls=":", lw=1.5, label=f"Falha Física ({T_fail_C:.0f} °C)")
ax_temp.plot(t_ms[-1], hist_Tmax_C[-1], marker="v", color="#B71C1C", markersize=8)

ax_temp.set_xlabel("Tempo Simulado (ms)", fontsize=9.5, fontweight="bold")
ax_temp.set_ylabel("Temperatura (°C)", fontsize=9.5, fontweight="bold")
ax_temp.set_title("Evolução Térmica (Aquecimento sem Cooler)", fontsize=10.5, fontweight="bold", color="#0D47A1")
ax_temp.legend(loc="upper left", fontsize=8, framealpha=0.92, facecolor="white")
ax_temp.set_ylim(20, max(hist_Tmax_C) * 1.10)

# -----------------------------------------------------------------------------
# PAINEL 4: RAZÃO n_i / N_d (CRITÉRIO EXATO DE FALHA DO SEMICONDUTOR)
# -----------------------------------------------------------------------------
ax_ni = fig.add_subplot(gs[1, 1])

ax_ni.semilogy(t_ms, hist_ni_ratio, color="#7B1FA2", lw=2.2, label="n_i(T) / N_d")
ax_ni.axhline(1.0, color="#C2185B", ls="--", lw=1.5, label="n_i = N_d (Colapso Lógico)")
ax_ni.axhline(0.01, color="#FB8C00", ls=":", lw=1.2, label="Início da Transição (1%)")
ax_ni.plot(t_ms[-1], hist_ni_ratio[-1], marker="v", color="#C2185B", markersize=8)

ax_ni.set_xlabel("Tempo Simulado (ms)", fontsize=9.5, fontweight="bold")
ax_ni.set_ylabel("Razão n_i / N_d (Escala Log)", fontsize=9.5, fontweight="bold")
ax_ni.set_title("Transição Extrínseco → Intrínseco", fontsize=10.5, fontweight="bold", color="#0D47A1")
ax_ni.legend(loc="lower right", fontsize=8, framealpha=0.92, facecolor="white")
ax_ni.set_ylim(min(hist_ni_ratio) * 0.5, 4.0)

# -----------------------------------------------------------------------------
# PAINEL 5: MOBILIDADE μ_n(T) E CONDUTIVIDADE σ(T)
# -----------------------------------------------------------------------------
ax_mob = fig.add_subplot(gs[1, 2])
col_m = "#1976D2"
col_s = "#E65100"

l1, = ax_mob.plot(t_ms, hist_mu_n_T, color=col_m, lw=2.2, label="μ_n [cm²/(V·s)]")
ax_mob.set_xlabel("Tempo Simulado (ms)", fontsize=9.5, fontweight="bold")
ax_mob.set_ylabel("Mobilidade μ_n [cm²/(V·s)]", color=col_m, fontsize=9.5, fontweight="bold")
ax_mob.tick_params(axis="y", labelcolor=col_m)

ax_sig = ax_mob.twinx()
ax_sig.set_facecolor("none")
l2, = ax_sig.plot(t_ms, hist_sigma_T, color=col_s, lw=2.0, ls="--", label="σ [S/cm]")
ax_sig.set_ylabel("Condutividade σ [S/cm]", color=col_s, fontsize=9.5, fontweight="bold")
ax_sig.tick_params(axis="y", labelcolor=col_s)

ax_mob.set_title("Degradação de Mobilidade e Condução", fontsize=10.5, fontweight="bold", color="#0D47A1")
ax_mob.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="center right", fontsize=8, framealpha=0.92, facecolor="white")

# Rodapé de resumo
fig.text(
    0.5, 0.015,
    f"Simulação concluída: O processador atinge o colapso lógico em t = {elapsed_us/1e3:.1f} ms | "
    f"T_máx = {T_max_final_C:.1f} °C | n_i(T) = {ni_final:.2e} cm^-3 ≥ N_d = {N_D:.1e} cm^-3 | "
    f"Os canais dos transistores perdem a capacidade de corte lógico devido à geração térmica maciça de pares e-h.",
    ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#37474F",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1", edgecolor="#B0BEC5", lw=1.0)
)

# Salva a imagem final em alta resolução
out_path = "simulacao_termica_resultado.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#F8F9FA")
plt.close(fig)

print(f"\n  ✓ Imagem gerada com sucesso em: {out_path}")
print("=" * 70)
