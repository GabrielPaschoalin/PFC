import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm


# ============================================================
# 1. CONFIGURAÇÕES GERAIS
# ============================================================

# Planta original:
# G(s) = Δx(s)/Δu(s)
# G(s) = -44.2989 / (s^3 + 32.74 s^2 - 1970.9327 s - 64235.88)

PLANT_GAIN_ORIGINAL = -44.2989

DEN_G = np.array([
    1.0,
    32.74,
    -1970.9327,
    -64235.88
])

# Usaremos a planta auxiliar positiva:
# H(s) = -G(s)
#
# IMPORTANTE:
# Projetar com H(s) significa que o comando encontrado no algoritmo
# é um comando auxiliar. Na implementação física, o sinal do controlador
# deve ser invertido para atuar corretamente sobre a planta original.
PLANT_GAIN = -PLANT_GAIN_ORIGINAL  # 44.2989


# Degrau de referência em variável de desvio.
# x positivo = esfera mais distante do núcleo.
# Como queremos a esfera mais próxima do núcleo: degrau negativo.
STEP_REF = -0.001  # [m] = -1 mm


# Critérios de desempenho
TS_MAX = 4.0                 # tempo de acomodação máximo [s]
SETTLING_BAND = 0.02         # critério de 2%
OVERSHOOT_MAX = 5.0          # [%]
ESS_MAX = 1e-5               # erro estacionário máximo [m]


# Limites físicos de tensão
V_MIN = 0.0
V_MAX = 16.84

# Parâmetros do ponto de operação
m = 0.022
g = 9.81
Kfm = 1.8632e-5
x0 = 0.01
R = 13.5

i0 = x0 * np.sqrt(m * g / Kfm)
u0 = R * i0

print(f"Corrente de operação i0 = {i0:.6f} A")
print(f"Tensão de operação u0 = {u0:.6f} V")
print(f"Margem inferior de Δu = {V_MIN - u0:.6f} V")
print(f"Margem superior de Δu = {V_MAX - u0:.6f} V")


# Tempo de simulação
T_FINAL = 10.0
N_TIME = 5000
t = np.linspace(0, T_FINAL, N_TIME)


# Busca grosseira com logspace
GRID_POINTS = 15

KP_RANGE = (1e2, 1e6)
KI_RANGE = (1e-1, 1e7)
KD_RANGE = (1e0, 1e5)


# Pesos da função custo
w1 = 1.0   # peso do tempo de acomodação
w2 = 2.0   # peso do overshoot
w3 = 5.0   # peso do erro estacionário
w4 = 0.5   # peso do esforço de controle


# Com PID ideal, o termo derivativo em degrau de referência gera um pico não físico.
# Por isso, neste primeiro teste, a saturação será calculada, mas não usada
# obrigatoriamente para rejeitar candidatos.
#
# Quando adicionarmos o polo da derivada filtrada, aí faz sentido colocar True.
REJECT_SATURATION = True

# Tempo inicial ignorado no cálculo do esforço de controle.
# Isso reduz o efeito do "derivative kick" do PID ideal.
EFFORT_SKIP_TIME = 0.02  # [s]


# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

def closed_loop_tf_coeffs(Kp, Ki, Kd):
    """
    Monta a malha fechada T(s) = C(s)H(s)/(1 + C(s)H(s)).

    H(s) = b / D(s)

    C(s) = Kd*s + Kp + Ki/s
         = (Kd*s^2 + Kp*s + Ki)/s

    Logo:

    T(s) = b*(Kd*s^2 + Kp*s + Ki)
           -----------------------------------------------
           s*D(s) + b*(Kd*s^2 + Kp*s + Ki)

    Retorna num, den.
    """

    b = PLANT_GAIN

    # Numerador: b*(Kd*s^2 + Kp*s + Ki)
    num = np.array([
        b * Kd,
        b * Kp,
        b * Ki
    ], dtype=float)

    # Denominador:
    # s*D(s) = s^4 + 32.74*s^3 -1970.9327*s^2 -64235.88*s
    #
    # somando b*(Kd*s^2 + Kp*s + Ki):
    den = np.array([
        1.0,
        DEN_G[1],
        DEN_G[2] + b * Kd,
        DEN_G[3] + b * Kp,
        b * Ki
    ], dtype=float)

    return num, den


def closed_loop_poles(Kp, Ki, Kd):
    _, den = closed_loop_tf_coeffs(Kp, Ki, Kd)
    return np.roots(den)


def is_stable(Kp, Ki, Kd):
    poles = closed_loop_poles(Kp, Ki, Kd)
    return np.all(np.real(poles) < 0)


def settling_time(y, ref, t, band=0.02):
    """
    Calcula o tempo de acomodação usando uma banda percentual
    em torno do valor final/ref.
    """

    tol = band * abs(ref)
    inside = np.abs(y - ref) <= tol

    for k in range(len(t)):
        if np.all(inside[k:]):
            return t[k]

    return np.nan


def overshoot_metrics(y, ref):
    """
    Calcula métricas de overshoot para degrau positivo ou negativo.

    Para ref < 0:
    - overshoot é quando y passa abaixo de ref, ou seja,
      aproxima mais do núcleo do que o valor desejado.

    Também calcula o pico no sentido oposto, útil para identificar
    respostas que primeiro se afastam do núcleo antes de aproximar.
    """

    amp = abs(ref)

    if ref < 0:
        overshoot_abs = max(0.0, ref - np.min(y))
        opposite_abs = max(0.0, np.max(y))
    else:
        overshoot_abs = max(0.0, np.max(y) - ref)
        opposite_abs = max(0.0, -np.min(y))

    overshoot_pct = 100.0 * overshoot_abs / amp
    opposite_pct = 100.0 * opposite_abs / amp

    peak_abs_pct = 100.0 * np.max(np.abs(y)) / amp

    return overshoot_pct, opposite_pct, peak_abs_pct


def simulate_step_response(Kp, Ki, Kd):
    """
    Simula a resposta ao degrau da malha fechada.
    """

    num, den = closed_loop_tf_coeffs(Kp, Ki, Kd)

    sys_cl = signal.TransferFunction(num, den)

    _, y_unit = signal.step(sys_cl, T=t)

    # Aplica o degrau desejado
    y = STEP_REF * y_unit

    return y


def estimate_control_effort(Kp, Ki, Kd, y):
    """
    Estima o esforço de controle usando:

    u_aux = Kp*e + Ki*integral(e) + Kd*de/dt

    Como projetamos com H(s) = -G(s), a tensão física real é:

    Δu_real = -u_aux

    e:

    u_real = u0 + Δu_real

    Observação importante:
    com PID ideal e degrau de referência, o termo derivativo gera
    um pico teórico infinito no instante inicial. Aqui o esforço é
    estimado numericamente e ignorando os primeiros milissegundos.
    """

    e = STEP_REF - y

    integral_e = cumulative_trapezoid(e, t, initial=0.0)
    derivative_e = np.gradient(e, t)

    u_aux = Kp * e + Ki * integral_e + Kd * derivative_e

    # Inversão necessária porque a planta original tem ganho negativo
    delta_u_real = -u_aux
    u_real = u0 + delta_u_real

    skip_idx = np.searchsorted(t, EFFORT_SKIP_TIME)

    delta_u_eval = delta_u_real[skip_idx:]
    u_eval = u_real[skip_idx:]

    max_abs_delta_u = np.max(np.abs(delta_u_eval))
    u_min_seen = np.min(u_eval)
    u_max_seen = np.max(u_eval)

    saturated = (u_min_seen < V_MIN) or (u_max_seen > V_MAX)

    return {
        "max_abs_delta_u": max_abs_delta_u,
        "u_min_seen": u_min_seen,
        "u_max_seen": u_max_seen,
        "saturated": saturated
    }


def evaluate_pid(Kp, Ki, Kd):
    """
    Avalia um candidato de PID.
    """

    poles = closed_loop_poles(Kp, Ki, Kd)

    if not np.all(np.real(poles) < 0):
        return None

    try:
        y = simulate_step_response(Kp, Ki, Kd)
    except Exception:
        return None

    if np.any(~np.isfinite(y)):
        return None

    # Erro estacionário estimado pela média dos últimos 5% da simulação
    n_tail = max(10, int(0.05 * len(y)))
    y_final_est = np.mean(y[-n_tail:])
    ess = abs(STEP_REF - y_final_est)

    Ts = settling_time(y, STEP_REF, t, SETTLING_BAND)

    overshoot_pct, opposite_pct, peak_abs_pct = overshoot_metrics(y, STEP_REF)

    effort = estimate_control_effort(Kp, Ki, Kd, y)

    # Critérios de aceitação
    accepted = True

    if not np.isfinite(Ts):
        accepted = False

    if np.isfinite(Ts) and Ts > TS_MAX:
        accepted = False

    if overshoot_pct > OVERSHOOT_MAX:
        accepted = False

    if ess > ESS_MAX:
        accepted = False

    if REJECT_SATURATION and effort["saturated"]:
        accepted = False

    # Função custo normalizada
    Ts_norm = Ts / TS_MAX if np.isfinite(Ts) else 1e6
    Mp_norm = overshoot_pct / OVERSHOOT_MAX
    ess_norm = ess / abs(STEP_REF)
    effort_norm = effort["max_abs_delta_u"] / (V_MAX - V_MIN)

    J = (
        w1 * Ts_norm
        + w2 * Mp_norm
        + w3 * ess_norm
        + w4 * effort_norm
    )

    return {
        "Kp": Kp,
        "Ki": Ki,
        "Kd": Kd,
        "J": J,
        "Ts": Ts,
        "ess": ess,
        "overshoot_pct": overshoot_pct,
        "opposite_pct": opposite_pct,
        "peak_abs_pct": peak_abs_pct,
        "max_abs_delta_u": effort["max_abs_delta_u"],
        "u_min_seen": effort["u_min_seen"],
        "u_max_seen": effort["u_max_seen"],
        "saturated": effort["saturated"],
        "accepted": accepted,
        "poles": poles
    }


# ============================================================
# 3. BUSCA GROSSEIRA
# ============================================================

def run_coarse_search():
    Kp_values = np.logspace(np.log10(KP_RANGE[0]), np.log10(KP_RANGE[1]), GRID_POINTS)
    Ki_values = np.logspace(np.log10(KI_RANGE[0]), np.log10(KI_RANGE[1]), GRID_POINTS)
    Kd_values = np.logspace(np.log10(KD_RANGE[0]), np.log10(KD_RANGE[1]), GRID_POINTS)

    total = len(Kp_values) * len(Ki_values) * len(Kd_values)

    results = []

    print(f"\nIniciando busca grosseira com {total} combinações...\n")

    for Kp in tqdm(Kp_values, desc="Kp"):
        for Ki in Ki_values:
            for Kd in Kd_values:
                result = evaluate_pid(Kp, Ki, Kd)
                if result is not None:
                    results.append(result)

    df = pd.DataFrame(results)

    if df.empty:
        print("Nenhum controlador estável foi encontrado.")
        return df

    df_sorted = df.sort_values("J").reset_index(drop=True)

    return df_sorted


# ============================================================
# 4. BUSCA REFINADA AO REDOR DOS MELHORES
# ============================================================

def run_refined_search(df_base, n_best=10, local_points=9, factor=5.0):
    """
    Refina a busca ao redor dos melhores candidatos encontrados.

    Para cada melhor candidato, cria uma nova malha logarítmica local
    em torno de Kp, Ki e Kd.
    """

    if df_base.empty:
        return df_base

    seeds = df_base.head(n_best)

    results = []

    print(f"\nIniciando busca refinada ao redor dos {n_best} melhores candidatos...\n")

    for _, row in tqdm(seeds.iterrows(), total=len(seeds), desc="Refino"):
        Kp0 = row["Kp"]
        Ki0 = row["Ki"]
        Kd0 = row["Kd"]

        Kp_values = np.logspace(np.log10(Kp0 / factor), np.log10(Kp0 * factor), local_points)
        Ki_values = np.logspace(np.log10(Ki0 / factor), np.log10(Ki0 * factor), local_points)
        Kd_values = np.logspace(np.log10(Kd0 / factor), np.log10(Kd0 * factor), local_points)

        for Kp in Kp_values:
            for Ki in Ki_values:
                for Kd in Kd_values:
                    result = evaluate_pid(Kp, Ki, Kd)
                    if result is not None:
                        results.append(result)

    df_refined = pd.DataFrame(results)

    if df_refined.empty:
        return df_base

    df_all = pd.concat([df_base, df_refined], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["Kp", "Ki", "Kd"])
    df_all = df_all.sort_values("J").reset_index(drop=True)

    return df_all


# ============================================================
# 5. PLOTS DO MELHOR CONTROLADOR
# ============================================================

def plot_best_candidate(best):
    Kp = best["Kp"]
    Ki = best["Ki"]
    Kd = best["Kd"]

    y = simulate_step_response(Kp, Ki, Kd)

    e = STEP_REF - y
    integral_e = cumulative_trapezoid(e, t, initial=0.0)
    derivative_e = np.gradient(e, t)

    u_aux = Kp * e + Ki * integral_e + Kd * derivative_e

    # Sinal físico real, por causa da planta original negativa
    delta_u_real = -u_aux
    u_real = u0 + delta_u_real

    poles = closed_loop_poles(Kp, Ki, Kd)

    # Zeros do controlador PID ideal:
    # Kd*s^2 + Kp*s + Ki = 0
    controller_zeros = np.roots([Kd, Kp, Ki])

    print("\n============================================================")
    print("MELHOR CONTROLADOR ENCONTRADO")
    print("============================================================")
    print(f"Kp = {Kp:.8g}")
    print(f"Ki = {Ki:.8g}")
    print(f"Kd = {Kd:.8g}")
    print(f"J  = {best['J']:.8g}")
    print(f"Ts = {best['Ts']:.6f} s")
    print(f"Erro estacionário = {best['ess']:.8e} m")
    print(f"Overshoot = {best['overshoot_pct']:.4f} %")
    print(f"Pico no sentido oposto = {best['opposite_pct']:.4f} %")
    print(f"Pico absoluto relativo = {best['peak_abs_pct']:.4f} %")
    print(f"Máximo |Δu| aproximado = {best['max_abs_delta_u']:.6f} V")
    print(f"u_min observado = {best['u_min_seen']:.6f} V")
    print(f"u_max observado = {best['u_max_seen']:.6f} V")
    print(f"Saturou? {best['saturated']}")
    print("\nPolos de malha fechada:")
    print(poles)
    print("\nZeros do controlador:")
    print(controller_zeros)

    plt.figure(figsize=(10, 5))
    plt.plot(t, y * 1000, label="Resposta Δx(t)")
    plt.axhline(STEP_REF * 1000, linestyle="--", label="Referência")
    plt.axhline((STEP_REF + SETTLING_BAND * abs(STEP_REF)) * 1000, linestyle=":", label="Banda 2%")
    plt.axhline((STEP_REF - SETTLING_BAND * abs(STEP_REF)) * 1000, linestyle=":")
    plt.grid(True)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Δx [mm]")
    plt.title("Resposta ao degrau de -1 mm")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(t, u_real, label="Tensão física estimada u(t)")
    plt.axhline(V_MIN, linestyle="--", label="Limite inferior")
    plt.axhline(V_MAX, linestyle="--", label="Limite superior")
    plt.axhline(u0, linestyle=":", label="u0")
    plt.grid(True)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensão [V]")
    plt.title("Esforço de controle estimado")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(poles), np.imag(poles), marker="x", s=100, label="Polos MF")
    plt.scatter(np.real(controller_zeros), np.imag(controller_zeros), marker="o", s=80, label="Zeros do PID")
    plt.axvline(0, color="black", linewidth=1)
    plt.axhline(0, color="black", linewidth=1)
    plt.grid(True)
    plt.xlabel("Parte real")
    plt.ylabel("Parte imaginária")
    plt.title("Polos de malha fechada e zeros do controlador")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    df = run_coarse_search()

    if not df.empty:
        df = run_refined_search(df, n_best=10, local_points=9, factor=5.0)

        # Salva todos os resultados
        df_to_save = df.drop(columns=["poles"], errors="ignore")
        df_to_save.to_csv("resultados_busca_pid_maglev.csv", index=False)

        print("\nArquivo salvo: resultados_busca_pid_maglev.csv")

        accepted = df[df["accepted"] == True].copy()

        print("\n============================================================")
        print("RESUMO")
        print("============================================================")
        print(f"Controladores estáveis avaliados: {len(df)}")
        print(f"Controladores aceitos pelos critérios: {len(accepted)}")

        if len(accepted) > 0:
            print("\nTop 10 controladores aceitos:")
            cols = [
                "Kp", "Ki", "Kd", "J", "Ts", "ess",
                "overshoot_pct", "opposite_pct",
                "max_abs_delta_u",
                "u_min_seen", "u_max_seen", "saturated"
            ]
            print(accepted[cols].head(10).to_string(index=False))

            best = accepted.iloc[0]
            plot_best_candidate(best)

        else:
            print("\nNenhum controlador atendeu todos os critérios.")
            print("Mostrando os 10 melhores controladores estáveis pelo custo J:")

            cols = [
                "Kp", "Ki", "Kd", "J", "Ts", "ess",
                "overshoot_pct", "opposite_pct",
                "max_abs_delta_u",
                "u_min_seen", "u_max_seen", "saturated"
            ]
            print(df[cols].head(10).to_string(index=False))

            best = df.iloc[0]
            plot_best_candidate(best)