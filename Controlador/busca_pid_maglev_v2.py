"""
busca_pid_maglev_v2.py

Busca automática de controlador PID para o Maglev linearizado.

Versão 2:
- Usa a planta auxiliar positiva H(s) = -G(s) para permitir varrer Kp, Ki e Kd positivos.
- Mantém comentários explícitos sobre a inversão de sinal necessária na planta física.
- Aplica degrau de -1 mm, isto é, esfera mais próxima do núcleo.
- Calcula erro estacionário, tempo de acomodação, overshoot, pico no sentido oposto,
  esforço de controle e saturação de tensão.
- Penaliza fortemente overshoot e saturação.
- Usa saturação como critério de rejeição.
- Inclui duas estratégias:
    1) Busca grosseira/refinada por ganhos usando logspace.
    2) Busca por polos desejados, gerando PID a partir da equação característica.
- Salva os resultados em CSV e plota o melhor candidato.

Dependências:
    pip install numpy scipy pandas matplotlib

Opcional:
    pip install tqdm

Autor: gerado para o projeto PFC - Maglev
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.integrate import cumulative_trapezoid


# ============================================================
# 0. PROGRESS BAR OPCIONAL
# ============================================================

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


warnings.filterwarnings("ignore", category=signal.BadCoefficients)


# ============================================================
# 1. CONFIGURAÇÕES GERAIS
# ============================================================

# ----------------------------
# Planta linearizada original
# ----------------------------
# G(s) = Δx(s)/Δu(s)
#
# G(s) = -44.2989 / (s^3 + 32.74 s^2 - 1970.9327 s - 64235.88)
#
# Como G(s) tem ganho negativo, a busca usa a planta auxiliar:
#
# H(s) = -G(s)
#
# Assim, é possível buscar Kp, Ki e Kd positivos usando realimentação padrão.
#
# IMPORTANTE PARA A IMPLEMENTAÇÃO FÍSICA:
# O controlador encontrado gera um comando auxiliar Δu_aux.
# Para aplicar na planta real, cuja FT tem ganho negativo, o comando físico deve ser:
#
# Δu_real = -Δu_aux
#
# Logo:
#
# u_real = u0 - Δu_aux

PLANT_GAIN_ORIGINAL = -44.2989
PLANT_GAIN = -PLANT_GAIN_ORIGINAL  # H(s) = -G(s), ganho positivo auxiliar

DEN_G = np.array([
    1.0,
    32.74,
    -1970.9327,
    -64235.88
], dtype=float)


# ----------------------------
# Degrau de referência
# ----------------------------
# x positivo = esfera mais distante do núcleo.
# x negativo = esfera mais próxima do núcleo.
#
# Teste solicitado:
# Δx_ref = -1 mm

STEP_REF = -0.001  # [m]


# ----------------------------
# Critérios de desempenho
# ----------------------------

TS_MAX = 4.0                 # [s] tempo de acomodação máximo
SETTLING_BAND = 0.02         # 2%
OVERSHOOT_MAX = 5.0          # [%]
ESS_MAX = 1e-5               # [m]


# ----------------------------
# Limites físicos de tensão
# ----------------------------

V_MIN = 0.0
V_MAX = 16.84


# ----------------------------
# Ponto de operação
# ----------------------------

m = 0.022
g = 9.81
Kfm = 1.8632e-5
x0 = 0.01
R = 13.5

i0 = x0 * math.sqrt(m * g / Kfm)
u0 = R * i0


# ----------------------------
# Tempo de simulação
# ----------------------------

T_FINAL = 10.0
N_TIME = 5000
t = np.linspace(0.0, T_FINAL, N_TIME)


# ----------------------------
# Modos de busca
# ----------------------------

RUN_GAIN_SEARCH = True
RUN_POLE_SEARCH = True
RUN_REFINED_SEARCH = True


# ----------------------------
# Busca por ganhos
# ----------------------------
# Dica:
# Para o sistema ficar estável, valores mínimos aproximados são:
#   Kd > 44.5
#   Kp > 1451
# mas a busca abaixo inclui uma margem maior.

GRID_POINTS = 13

KP_RANGE = (1e2, 2e5)
KI_RANGE = (1e-2, 2e5)
KD_RANGE = (1e1, 2e4)


# ----------------------------
# Busca refinada
# ----------------------------

REFINE_N_BEST = 12
REFINE_LOCAL_POINTS = 7
REFINE_FACTOR = 4.0


# ----------------------------
# Busca por polos desejados
# ----------------------------
# O PID ideal gera uma equação característica de 4ª ordem:
#
# s^4 + 32.74 s^3
# + (-1970.9327 + b Kd) s^2
# + (-64235.88 + b Kp) s
# + b Ki = 0
#
# O coeficiente de s^3 é fixo em 32.74.
# Portanto, a soma das "posições positivas" dos polos precisa ser:
#
# 2*sigma + p3 + p4 = 32.74
#
# onde os polos são:
#   -sigma ± j wd
#   -p3
#   -p4

POLE_ZETA_VALUES = np.linspace(0.70, 0.95, 11)   # zeta >= ~0.69 para overshoot menor que 5%
POLE_SIGMA_VALUES = np.linspace(1.0, 5.5, 19)    # sigma ~ 4/Ts; sigma >= 1 dá Ts <= 4
POLE_P3_VALUES = np.linspace(2.0, 28.0, 53)      # p4 vem da restrição da soma


# ----------------------------
# Critério de saturação
# ----------------------------
# Se True, controladores que saturam são rejeitados.
# Para este caso, isso é importante porque o ponto de operação u0 está perto de V_MAX.

REJECT_SATURATION = True


# ----------------------------
# Esforço de controle
# ----------------------------
# Com PID ideal, derivada no erro gera "derivative kick" no degrau de referência.
# Para estimar esforço de controle de forma mais próxima do que se implementaria
# no microcontrolador, usa-se derivada na medição:
#
#   u_aux = Kp*e + Ki*integral(e) - Kd*dy/dt
#
# Isso mantém a ação derivativa contra variações da saída, mas evita derivar o degrau
# da referência.
#
# Caso queira estimar a forma ideal pura, troque para:
#   EFFORT_DERIVATIVE_MODE = "error"

EFFORT_DERIVATIVE_MODE = "measurement"  # "measurement" ou "error"
EFFORT_SKIP_TIME = 0.02                 # [s], usado apenas para ignorar transiente numérico inicial


# ----------------------------
# Pesos da função custo
# ----------------------------
# Em relação à V1:
# - overshoot recebeu peso muito maior;
# - esforço também recebeu peso maior;
# - saturação recebe penalização pesada adicional.

w_ts = 1.0
w_overshoot = 20.0
w_ess = 10.0
w_effort = 5.0

PENALTY_OVERSHOOT = 200.0
PENALTY_TS = 50.0
PENALTY_ESS = 50.0
PENALTY_SATURATION = 1000.0
PENALTY_OPPOSITE_PEAK = 5.0


# ----------------------------
# Arquivos de saída
# ----------------------------

OUTPUT_DIR = Path(".")
CSV_ALL = OUTPUT_DIR / "resultados_busca_pid_maglev_v2_todos.csv"
CSV_ACCEPTED = OUTPUT_DIR / "resultados_busca_pid_maglev_v2_aceitos.csv"


# ============================================================
# 2. ESTRUTURAS DE DADOS
# ============================================================

@dataclass
class PIDCandidate:
    Kp: float
    Ki: float
    Kd: float
    source: str


# ============================================================
# 3. FUNÇÕES DO MODELO
# ============================================================

def closed_loop_tf_coeffs(Kp: float, Ki: float, Kd: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Monta a malha fechada T(s) = C(s)H(s)/(1 + C(s)H(s)).

    H(s) = b / D(s)

    C(s) = Kd*s + Kp + Ki/s
         = (Kd*s^2 + Kp*s + Ki)/s

    T(s) = b*(Kd*s^2 + Kp*s + Ki)
           -----------------------------------------------
           s*D(s) + b*(Kd*s^2 + Kp*s + Ki)

    Retorna:
        num, den
    """

    b = PLANT_GAIN

    num = np.array([
        b * Kd,
        b * Kp,
        b * Ki
    ], dtype=float)

    den = np.array([
        1.0,
        DEN_G[1],
        DEN_G[2] + b * Kd,
        DEN_G[3] + b * Kp,
        b * Ki
    ], dtype=float)

    return trim_leading_zeros(num), trim_leading_zeros(den)


def trim_leading_zeros(poly: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    idx = 0
    while idx < len(poly) - 1 and abs(poly[idx]) < tol:
        idx += 1
    return poly[idx:]


def closed_loop_poles(Kp: float, Ki: float, Kd: float) -> np.ndarray:
    _, den = closed_loop_tf_coeffs(Kp, Ki, Kd)
    return np.roots(den)


def controller_zeros(Kp: float, Ki: float, Kd: float) -> np.ndarray:
    # Zeros do numerador do PID ideal: Kd*s^2 + Kp*s + Ki
    if abs(Kd) > 1e-14:
        return np.roots([Kd, Kp, Ki])
    if abs(Kp) > 1e-14:
        return np.roots([Kp, Ki])
    return np.array([])


def simulate_step_response(Kp: float, Ki: float, Kd: float) -> np.ndarray | None:
    """
    Simula a resposta ao degrau de referência STEP_REF.
    """
    try:
        num, den = closed_loop_tf_coeffs(Kp, Ki, Kd)
        sys_cl = signal.TransferFunction(num, den)
        _, y_unit = signal.step(sys_cl, T=t)
        y = STEP_REF * y_unit
        if np.any(~np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


# ============================================================
# 4. MÉTRICAS
# ============================================================

def settling_time(y: np.ndarray, ref: float, time: np.ndarray, band: float = 0.02) -> float:
    """
    Tempo de acomodação:
    primeiro instante em que a resposta entra e permanece dentro da banda.
    """
    tol = band * abs(ref)
    inside = np.abs(y - ref) <= tol

    for k in range(len(time)):
        if np.all(inside[k:]):
            return float(time[k])

    return float("nan")


def overshoot_metrics(y: np.ndarray, ref: float) -> tuple[float, float, float]:
    """
    Calcula:
    - overshoot_pct: ultrapassagem no sentido da referência;
    - opposite_pct: pico no sentido oposto;
    - peak_abs_pct: maior pico absoluto relativo ao degrau.

    Para ref < 0:
        overshoot ocorre quando y < ref.
        pico oposto ocorre quando y > 0.
    """

    amp = abs(ref)

    if ref < 0:
        overshoot_abs = max(0.0, ref - float(np.min(y)))
        opposite_abs = max(0.0, float(np.max(y)))
    else:
        overshoot_abs = max(0.0, float(np.max(y)) - ref)
        opposite_abs = max(0.0, -float(np.min(y)))

    overshoot_pct = 100.0 * overshoot_abs / amp
    opposite_pct = 100.0 * opposite_abs / amp
    peak_abs_pct = 100.0 * float(np.max(np.abs(y))) / amp

    return overshoot_pct, opposite_pct, peak_abs_pct


def estimate_control_effort(Kp: float, Ki: float, Kd: float, y: np.ndarray) -> dict:
    """
    Estima o esforço de controle.

    Como a busca usa H(s) = -G(s), o sinal físico real deve ser invertido:

        Δu_real = -Δu_aux

    e:

        u_real = u0 + Δu_real

    Modos:
        measurement:
            u_aux = Kp*e + Ki*int(e) - Kd*dy/dt

        error:
            u_aux = Kp*e + Ki*int(e) + Kd*de/dt
    """

    e = STEP_REF - y
    integral_e = cumulative_trapezoid(e, t, initial=0.0)

    if EFFORT_DERIVATIVE_MODE == "measurement":
        derivative_term = -np.gradient(y, t)
    elif EFFORT_DERIVATIVE_MODE == "error":
        derivative_term = np.gradient(e, t)
    else:
        raise ValueError("EFFORT_DERIVATIVE_MODE deve ser 'measurement' ou 'error'.")

    u_aux = Kp * e + Ki * integral_e + Kd * derivative_term

    # Inversão por causa da planta real negativa
    delta_u_real = -u_aux
    u_real = u0 + delta_u_real

    skip_idx = np.searchsorted(t, EFFORT_SKIP_TIME)

    delta_u_eval = delta_u_real[skip_idx:]
    u_eval = u_real[skip_idx:]

    max_abs_delta_u = float(np.max(np.abs(delta_u_eval)))
    u_min_seen = float(np.min(u_eval))
    u_max_seen = float(np.max(u_eval))

    saturated = (u_min_seen < V_MIN) or (u_max_seen > V_MAX)

    return {
        "max_abs_delta_u": max_abs_delta_u,
        "u_min_seen": u_min_seen,
        "u_max_seen": u_max_seen,
        "saturated": bool(saturated),
        "u_real": u_real,
        "delta_u_real": delta_u_real,
        "u_aux": u_aux,
    }


def evaluate_pid(candidate: PIDCandidate) -> dict | None:
    """
    Avalia um PID candidato.
    """

    Kp = float(candidate.Kp)
    Ki = float(candidate.Ki)
    Kd = float(candidate.Kd)

    if Kp <= 0 or Ki <= 0 or Kd < 0:
        return None

    poles = closed_loop_poles(Kp, Ki, Kd)

    if np.any(~np.isfinite(poles)):
        return None

    stable = bool(np.all(np.real(poles) < 0))

    if not stable:
        return None

    y = simulate_step_response(Kp, Ki, Kd)
    if y is None:
        return None

    n_tail = max(10, int(0.05 * len(y)))
    y_final_est = float(np.mean(y[-n_tail:]))
    ess = abs(STEP_REF - y_final_est)

    Ts = settling_time(y, STEP_REF, t, SETTLING_BAND)
    overshoot_pct, opposite_pct, peak_abs_pct = overshoot_metrics(y, STEP_REF)

    effort = estimate_control_effort(Kp, Ki, Kd, y)

    saturated = bool(effort["saturated"])

    accepted = True

    if not np.isfinite(Ts) or Ts > TS_MAX:
        accepted = False

    if overshoot_pct > OVERSHOOT_MAX:
        accepted = False

    if ess > ESS_MAX:
        accepted = False

    if REJECT_SATURATION and saturated:
        accepted = False

    # Custo base normalizado
    Ts_norm = (Ts / TS_MAX) if np.isfinite(Ts) else 100.0
    Mp_norm = overshoot_pct / OVERSHOOT_MAX
    ess_norm = ess / max(abs(STEP_REF), 1e-12)
    effort_norm = effort["max_abs_delta_u"] / max((V_MAX - V_MIN), 1e-12)

    J = (
        w_ts * Ts_norm
        + w_overshoot * Mp_norm
        + w_ess * ess_norm
        + w_effort * effort_norm
    )

    # Penalizações fortes por violação
    if np.isfinite(Ts) and Ts > TS_MAX:
        J += PENALTY_TS * ((Ts - TS_MAX) / TS_MAX) ** 2
    elif not np.isfinite(Ts):
        J += PENALTY_TS * 100.0

    if overshoot_pct > OVERSHOOT_MAX:
        J += PENALTY_OVERSHOOT * ((overshoot_pct - OVERSHOOT_MAX) / OVERSHOOT_MAX) ** 2

    if ess > ESS_MAX:
        J += PENALTY_ESS * ((ess - ESS_MAX) / ESS_MAX) ** 2

    if saturated:
        lower_violation = max(0.0, V_MIN - effort["u_min_seen"])
        upper_violation = max(0.0, effort["u_max_seen"] - V_MAX)
        sat_violation = lower_violation + upper_violation
        J += PENALTY_SATURATION * (1.0 + sat_violation / max((V_MAX - V_MIN), 1e-12))

    # Pequena penalização para resposta inicial no sentido oposto
    J += PENALTY_OPPOSITE_PEAK * (opposite_pct / 100.0)

    return {
        "Kp": Kp,
        "Ki": Ki,
        "Kd": Kd,
        "source": candidate.source,
        "J": float(J),
        "stable": stable,
        "accepted": bool(accepted),
        "Ts": float(Ts),
        "ess": float(ess),
        "overshoot_pct": float(overshoot_pct),
        "opposite_pct": float(opposite_pct),
        "peak_abs_pct": float(peak_abs_pct),
        "max_abs_delta_u": float(effort["max_abs_delta_u"]),
        "u_min_seen": float(effort["u_min_seen"]),
        "u_max_seen": float(effort["u_max_seen"]),
        "saturated": bool(saturated),
        "poles": poles,
        "zeros": controller_zeros(Kp, Ki, Kd),
    }


# ============================================================
# 5. GERAÇÃO DE CANDIDATOS
# ============================================================

def generate_gain_candidates() -> list[PIDCandidate]:
    """
    Gera candidatos por varredura logarítmica de Kp, Ki e Kd.
    """

    Kp_values = np.logspace(np.log10(KP_RANGE[0]), np.log10(KP_RANGE[1]), GRID_POINTS)
    Ki_values = np.logspace(np.log10(KI_RANGE[0]), np.log10(KI_RANGE[1]), GRID_POINTS)
    Kd_values = np.logspace(np.log10(KD_RANGE[0]), np.log10(KD_RANGE[1]), GRID_POINTS)

    candidates = []

    for Kp in Kp_values:
        for Ki in Ki_values:
            for Kd in Kd_values:
                candidates.append(PIDCandidate(Kp=Kp, Ki=Ki, Kd=Kd, source="gain_logspace"))

    return candidates


def gains_from_desired_polynomial(coeffs: np.ndarray) -> tuple[float, float, float]:
    """
    Dado o polinômio desejado:

        s^4 + a3 s^3 + a2 s^2 + a1 s + a0

    calcula os ganhos do PID ideal usando:

        a2 = DEN_G[2] + b*Kd
        a1 = DEN_G[3] + b*Kp
        a0 = b*Ki

    Como a3 é fixo pela planta, esta função assume que o polinômio
    já foi construído com a3 ~= 32.74.
    """

    b = PLANT_GAIN

    a2 = coeffs[2]
    a1 = coeffs[3]
    a0 = coeffs[4]

    Kd = (a2 - DEN_G[2]) / b
    Kp = (a1 - DEN_G[3]) / b
    Ki = a0 / b

    return float(Kp), float(Ki), float(Kd)


def generate_pole_candidates() -> list[PIDCandidate]:
    """
    Gera candidatos por alocação paramétrica de polos.

    Polos desejados:
        -sigma ± j wd
        -p3
        -p4

    com:
        2*sigma + p3 + p4 = 32.74

    para respeitar o coeficiente fixo de s^3.
    """

    candidates = []

    a3_fixed = DEN_G[1]  # 32.74

    for zeta in POLE_ZETA_VALUES:
        for sigma in POLE_SIGMA_VALUES:
            if sigma <= 0 or zeta <= 0 or zeta >= 1:
                continue

            wn = sigma / zeta
            wd_sq = max(0.0, wn**2 - sigma**2)
            wd = math.sqrt(wd_sq)

            for p3 in POLE_P3_VALUES:
                p4 = a3_fixed - 2.0 * sigma - p3

                if p4 <= 0:
                    continue

                poles = np.array([
                    complex(-sigma, wd),
                    complex(-sigma, -wd),
                    complex(-p3, 0.0),
                    complex(-p4, 0.0),
                ])

                coeffs = np.poly(poles).real

                # Garante que o coeficiente de s^3 respeita a restrição
                if abs(coeffs[1] - a3_fixed) > 1e-6:
                    continue

                Kp, Ki, Kd = gains_from_desired_polynomial(coeffs)

                if Kp > 0 and Ki > 0 and Kd >= 0:
                    candidates.append(PIDCandidate(Kp=Kp, Ki=Ki, Kd=Kd, source="pole_search"))

    return candidates


def generate_refined_candidates(df_base: pd.DataFrame) -> list[PIDCandidate]:
    """
    Refina ao redor dos melhores candidatos encontrados.
    """

    if df_base.empty:
        return []

    seeds = df_base.head(REFINE_N_BEST)

    candidates = []

    for _, row in seeds.iterrows():
        Kp0 = float(row["Kp"])
        Ki0 = float(row["Ki"])
        Kd0 = float(row["Kd"])

        if Kp0 <= 0 or Ki0 <= 0 or Kd0 <= 0:
            continue

        Kp_values = np.logspace(np.log10(Kp0 / REFINE_FACTOR), np.log10(Kp0 * REFINE_FACTOR), REFINE_LOCAL_POINTS)
        Ki_values = np.logspace(np.log10(Ki0 / REFINE_FACTOR), np.log10(Ki0 * REFINE_FACTOR), REFINE_LOCAL_POINTS)
        Kd_values = np.logspace(np.log10(Kd0 / REFINE_FACTOR), np.log10(Kd0 * REFINE_FACTOR), REFINE_LOCAL_POINTS)

        for Kp in Kp_values:
            for Ki in Ki_values:
                for Kd in Kd_values:
                    candidates.append(PIDCandidate(Kp=Kp, Ki=Ki, Kd=Kd, source="refined"))

    return candidates


def evaluate_candidates(candidates: list[PIDCandidate], description: str) -> pd.DataFrame:
    """
    Avalia lista de candidatos.
    """

    results = []

    print(f"\n{description}")
    print(f"Total de candidatos: {len(candidates)}")

    for cand in tqdm(candidates):
        result = evaluate_pid(cand)
        if result is not None:
            results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Remove duplicatas numéricas simples
    df["Kp_round"] = df["Kp"].round(10)
    df["Ki_round"] = df["Ki"].round(10)
    df["Kd_round"] = df["Kd"].round(10)

    df = df.drop_duplicates(subset=["Kp_round", "Ki_round", "Kd_round"])
    df = df.drop(columns=["Kp_round", "Ki_round", "Kd_round"])

    df = df.sort_values("J").reset_index(drop=True)

    return df


# ============================================================
# 6. PLOTS E RELATÓRIOS
# ============================================================

def recompute_response_and_effort(row: pd.Series) -> tuple[np.ndarray, dict]:
    Kp = float(row["Kp"])
    Ki = float(row["Ki"])
    Kd = float(row["Kd"])

    y = simulate_step_response(Kp, Ki, Kd)
    effort = estimate_control_effort(Kp, Ki, Kd, y)

    return y, effort


def print_project_context() -> None:
    print("\n============================================================")
    print("CONTEXTO DO PROJETO")
    print("============================================================")
    print("Planta original:")
    print("G(s) = -44.2989 / (s^3 + 32.74 s^2 - 1970.9327 s - 64235.88)")
    print("\nPlanta usada na busca:")
    print("H(s) = -G(s)")
    print("\nImplicação:")
    print("Os ganhos encontrados são para a planta auxiliar H.")
    print("Na implementação física, use Δu_real = -Δu_aux.")
    print("\nPonto de operação:")
    print(f"i0 = {i0:.6f} A")
    print(f"u0 = {u0:.6f} V")
    print(f"Limite de Δu inferior = {V_MIN - u0:.6f} V")
    print(f"Limite de Δu superior = {V_MAX - u0:.6f} V")
    print("\nDegrau:")
    print(f"STEP_REF = {STEP_REF:.6f} m = {STEP_REF*1000:.3f} mm")
    print("\nCritérios:")
    print(f"Ts <= {TS_MAX:.3f} s")
    print(f"Overshoot <= {OVERSHOOT_MAX:.3f} %")
    print(f"ess <= {ESS_MAX:.3e} m")
    print(f"Saturação rejeitada? {REJECT_SATURATION}")
    print(f"Modo de derivada no esforço: {EFFORT_DERIVATIVE_MODE}")


def print_best(row: pd.Series, title: str) -> None:
    poles = row["poles"]
    zeros = row["zeros"]

    print("\n============================================================")
    print(title)
    print("============================================================")
    print(f"Origem do candidato: {row['source']}")
    print(f"Kp = {row['Kp']:.10g}")
    print(f"Ki = {row['Ki']:.10g}")
    print(f"Kd = {row['Kd']:.10g}")
    print(f"J  = {row['J']:.10g}")
    print(f"Aceito? {bool(row['accepted'])}")
    print(f"Ts = {row['Ts']:.6f} s")
    print(f"Erro estacionário = {row['ess']:.8e} m")
    print(f"Overshoot = {row['overshoot_pct']:.4f} %")
    print(f"Pico no sentido oposto = {row['opposite_pct']:.4f} %")
    print(f"Pico absoluto relativo = {row['peak_abs_pct']:.4f} %")
    print(f"Máximo |Δu| aproximado = {row['max_abs_delta_u']:.6f} V")
    print(f"u_min observado = {row['u_min_seen']:.6f} V")
    print(f"u_max observado = {row['u_max_seen']:.6f} V")
    print(f"Saturou? {bool(row['saturated'])}")

    print("\nPolos de malha fechada:")
    print(poles)

    print("\nZeros do controlador PID:")
    print(zeros)


def plot_candidate(row: pd.Series, suffix: str = "") -> None:
    y, effort = recompute_response_and_effort(row)

    Kp = float(row["Kp"])
    Ki = float(row["Ki"])
    Kd = float(row["Kd"])

    poles = closed_loop_poles(Kp, Ki, Kd)
    zeros = controller_zeros(Kp, Ki, Kd)

    u_real = effort["u_real"]
    delta_u_real = effort["delta_u_real"]

    # Resposta
    plt.figure(figsize=(10, 5))
    plt.plot(t, y * 1000.0, label="Resposta Δx(t)")
    plt.axhline(STEP_REF * 1000.0, linestyle="--", label="Referência")
    plt.axhline((STEP_REF + SETTLING_BAND * abs(STEP_REF)) * 1000.0, linestyle=":", label="Banda 2%")
    plt.axhline((STEP_REF - SETTLING_BAND * abs(STEP_REF)) * 1000.0, linestyle=":")
    plt.grid(True)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Δx [mm]")
    plt.title(f"Resposta ao degrau de {STEP_REF*1000:.1f} mm {suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Tensão absoluta
    plt.figure(figsize=(10, 5))
    plt.plot(t, u_real, label="Tensão física estimada u(t)")
    plt.axhline(V_MIN, linestyle="--", label="Limite inferior")
    plt.axhline(V_MAX, linestyle="--", label="Limite superior")
    plt.axhline(u0, linestyle=":", label="u0")
    plt.grid(True)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensão [V]")
    plt.title(f"Esforço de controle estimado {suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Delta u
    plt.figure(figsize=(10, 5))
    plt.plot(t, delta_u_real, label="Δu_real(t)")
    plt.axhline(V_MIN - u0, linestyle="--", label="Δu mínimo")
    plt.axhline(V_MAX - u0, linestyle="--", label="Δu máximo")
    plt.axhline(0.0, linestyle=":", label="0")
    plt.grid(True)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Δu [V]")
    plt.title(f"Comando em variável de desvio {suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Polos e zeros
    plt.figure(figsize=(7, 6))
    plt.scatter(np.real(poles), np.imag(poles), marker="x", s=100, label="Polos MF")
    if len(zeros) > 0:
        plt.scatter(np.real(zeros), np.imag(zeros), marker="o", s=80, label="Zeros PID")
    plt.axvline(0, color="black", linewidth=1)
    plt.axhline(0, color="black", linewidth=1)
    plt.grid(True)
    plt.xlabel("Parte real")
    plt.ylabel("Parte imaginária")
    plt.title(f"Polos de malha fechada e zeros do PID {suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_results(df: pd.DataFrame) -> None:
    if df.empty:
        return

    # Não dá para salvar arrays complexos diretamente de forma elegante,
    # então cria strings para polos e zeros.
    df_save = df.copy()
    df_save["poles"] = df_save["poles"].apply(lambda arr: "; ".join([str(z) for z in arr]))
    df_save["zeros"] = df_save["zeros"].apply(lambda arr: "; ".join([str(z) for z in arr]))

    df_save.to_csv(CSV_ALL, index=False)

    accepted = df_save[df_save["accepted"] == True].copy()
    accepted.to_csv(CSV_ACCEPTED, index=False)

    print("\nArquivos salvos:")
    print(f"- {CSV_ALL}")
    print(f"- {CSV_ACCEPTED}")


# ============================================================
# 7. EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    print_project_context()

    dataframes = []

    if RUN_GAIN_SEARCH:
        gain_candidates = generate_gain_candidates()
        df_gain = evaluate_candidates(gain_candidates, "Busca 1: varredura logspace em Kp, Ki e Kd")
        if not df_gain.empty:
            dataframes.append(df_gain)

    if RUN_POLE_SEARCH:
        pole_candidates = generate_pole_candidates()
        df_pole = evaluate_candidates(pole_candidates, "Busca 2: varredura por polos desejados")
        if not df_pole.empty:
            dataframes.append(df_pole)

    if not dataframes:
        print("\nNenhum controlador estável foi encontrado.")
        return

    df_all = pd.concat(dataframes, ignore_index=True)
    df_all = df_all.sort_values("J").reset_index(drop=True)

    if RUN_REFINED_SEARCH:
        refined_candidates = generate_refined_candidates(df_all)
        df_refined = evaluate_candidates(refined_candidates, "Busca 3: refino local ao redor dos melhores")
        if not df_refined.empty:
            df_all = pd.concat([df_all, df_refined], ignore_index=True)
            df_all = df_all.sort_values("J").reset_index(drop=True)

    # Remove duplicatas após juntar tudo
    df_all["Kp_round"] = df_all["Kp"].round(10)
    df_all["Ki_round"] = df_all["Ki"].round(10)
    df_all["Kd_round"] = df_all["Kd"].round(10)
    df_all = df_all.drop_duplicates(subset=["Kp_round", "Ki_round", "Kd_round"])
    df_all = df_all.drop(columns=["Kp_round", "Ki_round", "Kd_round"])
    df_all = df_all.sort_values("J").reset_index(drop=True)

    accepted = df_all[df_all["accepted"] == True].copy()

    save_results(df_all)

    print("\n============================================================")
    print("RESUMO")
    print("============================================================")
    print(f"Controladores estáveis avaliados: {len(df_all)}")
    print(f"Controladores aceitos pelos critérios: {len(accepted)}")

    cols = [
        "source", "Kp", "Ki", "Kd", "J", "accepted",
        "Ts", "ess", "overshoot_pct", "opposite_pct",
        "max_abs_delta_u", "u_min_seen", "u_max_seen", "saturated"
    ]

    if len(accepted) > 0:
        print("\nTop 10 controladores aceitos:")
        print(accepted[cols].head(10).to_string(index=False))

        best = accepted.iloc[0]
        print_best(best, "MELHOR CONTROLADOR ACEITO")
        plot_candidate(best, suffix="(melhor aceito)")

    else:
        print("\nNenhum controlador atendeu todos os critérios.")
        print("Mostrando os 10 melhores controladores estáveis, já com penalizações por violação:")
        print(df_all[cols].head(10).to_string(index=False))

        best = df_all.iloc[0]
        print_best(best, "MELHOR CONTROLADOR ESTÁVEL, MAS NÃO ACEITO")
        plot_candidate(best, suffix="(melhor não aceito)")


if __name__ == "__main__":
    main()
