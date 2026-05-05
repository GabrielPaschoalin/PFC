"""
validacao_maglev_v9.py

Validação local do modelo linearizado do levitador eletromagnético.

Baseada na v8, com ajuste visual:
- as simulações continuam em unidades do SI (metros);
- os gráficos de posição são exibidos em centímetros para facilitar a leitura.

Dependências:
    pip install numpy scipy matplotlib
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import solve_ivp
from scipy import signal


PASTA_SAIDA = Path(__file__).resolve().parent


# ============================================================
# Parâmetros
# ============================================================

m = 0.022
g = 9.81
R = 13.5
Kfm = 1.8632e-5
kL = 0.0000415
Lb = 0.4082238

x0 = 0.01
i0 = x0 * np.sqrt(m * g / Kfm)
v0 = 0.0
u0 = R * i0

X0 = np.array([i0, x0, v0])


# ============================================================
# Configuração dos gráficos
# ============================================================

FATOR_POSICAO = 100.0     # conversão de m para cm apenas para os gráficos
UNIDADE_POSICAO = "cm"


# ============================================================
# Funções do modelo não linear
# ============================================================

def L(x):
    return kL / x + Lb


def Fm(i, x):
    return Kfm * i**2 / x**2


def Bl(i, x):
    return Kfm * i / x**2


def modelo_nao_linear(t, X, delta_u):
    i, x, v = X
    x = max(x, 1e-6)

    u = u0 + delta_u(t)

    di = (u - R * i - Bl(i, x) * v) / L(x)
    dx = v
    dv = g - Fm(i, x) / m

    return [di, dx, dv]


# ============================================================
# Modelo linearizado
# ============================================================

den = kL + Lb * x0

A = np.array([
    [-R * x0 / den, 0.0, -Kfm * i0 / (x0 * den)],
    [0.0, 0.0, 1.0],
    [-2 * Kfm * i0 / (m * x0**2), 2 * Kfm * i0**2 / (m * x0**3), 0.0]
])

B = np.array([[x0 / den], [0.0], [0.0]])
C = np.array([[0.0, 1.0, 0.0]])
D = np.array([[0.0]])

sistema_linear = signal.StateSpace(A, B, C, D)


# ============================================================
# Gráficos
# ============================================================

def salvar_grafico_png(
    t,
    x_nl,
    x_lin,
    titulo,
    nome_base,
    ylim=None,
    casas_decimais_y=None,
    event_time=None,
    reference_signal=None,
    reference_label=None
):
    caminho_png = PASTA_SAIDA / f"{nome_base}.png"

    # Conversão somente para visualização.
    x_nl_plot = x_nl * FATOR_POSICAO
    x_lin_plot = x_lin * FATOR_POSICAO

    if reference_signal is not None:
        reference_signal_plot = reference_signal * FATOR_POSICAO
    else:
        reference_signal_plot = None

    fig, ax = plt.subplots(figsize=(8, 5))

    if event_time is not None:
        ax.axvline(
            event_time,
            color="black",
            linestyle="-",
            linewidth=2.0,
            label="Instante da mudança"
        )

    if reference_signal_plot is not None:
        ax.step(
            t,
            reference_signal_plot,
            where="post",
            color="black",
            linestyle="--",
            linewidth=1.4,
            label=reference_label if reference_label else "Referência"
        )

    ax.plot(t, x_nl_plot, label="Modelo não linear")
    ax.plot(t, x_lin_plot, "--", label="Modelo linearizado")

    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel(f"Posição x [{UNIDADE_POSICAO}]")
    ax.set_title(titulo)
    ax.grid(True)

    # Remove a folga automática do Matplotlib no eixo do tempo,
    # fazendo o gráfico começar exatamente em t = 0.
    ax.set_xlim(t[0], t[-1])
    ax.margins(x=0)

    ax.legend()

    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    if ylim is not None:
        ax.set_ylim(ylim)

    if casas_decimais_y is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{casas_decimais_y}f"))

    fig.tight_layout()
    fig.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return caminho_png


# ============================================================
# Simulações
# ============================================================

def simular(
    t,
    delta_X0,
    delta_u,
    titulo,
    nome_base,
    ylim=None,
    casas_decimais_y=None,
    event_time=None,
    reference_signal=None,
    reference_label=None
):
    sol_nl = solve_ivp(
        fun=lambda tau, X: modelo_nao_linear(tau, X, delta_u),
        t_span=(t[0], t[-1]),
        y0=X0 + delta_X0,
        t_eval=t,
        rtol=1e-8,
        atol=1e-10,
        max_step=1e-5
    )

    if not sol_nl.success:
        raise RuntimeError(f"Erro na simulação não linear: {sol_nl.message}")

    u_lin = np.array([delta_u(ti) for ti in t])
    _, y_lin_delta, _ = signal.lsim(
        sistema_linear,
        U=u_lin,
        T=t,
        X0=delta_X0
    )

    y_lin_delta = np.asarray(y_lin_delta).reshape(-1)

    x_nl = sol_nl.y[1]
    x_lin = x0 + y_lin_delta

    rmse = np.sqrt(np.mean((x_nl - x_lin)**2))

    caminho_png = salvar_grafico_png(
        t=t,
        x_nl=x_nl,
        x_lin=x_lin,
        titulo=titulo,
        nome_base=nome_base,
        ylim=ylim,
        casas_decimais_y=casas_decimais_y,
        event_time=event_time,
        reference_signal=reference_signal,
        reference_label=reference_label,
    )

    print(f"{titulo}")
    print(f"  RMSE = {rmse:.6e} m")
    print(f"  RMSE = {rmse * FATOR_POSICAO:.6e} cm")
    print(f"  PNG salvo em: {caminho_png}\n")


def simular_perturbacao_posicao_em_tempo(
    t,
    t_pert,
    delta_x,
    titulo,
    nome_base,
    reference_label="Referência de posição"
):
    """
    Aplica uma perturbação instantânea na posição em t = t_pert.

    A referência mostrada no gráfico é:
        ref(t) = x0,                 para t < t_pert
        ref(t) = x0 + delta_x,       para t >= t_pert
    """
    delta_u = lambda tau: 0.0

    if not (t[0] < t_pert < t[-1]):
        raise ValueError("t_pert deve estar dentro do intervalo de simulação.")

    t_final = t[-1]

    n_total = len(t)
    n_pre = max(2, int(n_total * (t_pert - t[0]) / (t_final - t[0])))
    n_pos = max(2, n_total - n_pre + 1)

    t_pre_abs = np.linspace(t[0], t_pert, n_pre)
    t_pos_abs = np.linspace(t_pert, t_final, n_pos)

    tau_pre = np.linspace(0.0, t_pert - t[0], n_pre)
    tau_pos = np.linspace(0.0, t_final - t_pert, n_pos)

    # Não linear
    sol_pre_nl = solve_ivp(
        fun=lambda tau, X: modelo_nao_linear(tau, X, delta_u),
        t_span=(t_pre_abs[0], t_pre_abs[-1]),
        y0=X0,
        t_eval=t_pre_abs,
        rtol=1e-8,
        atol=1e-10,
        max_step=1e-5
    )

    if not sol_pre_nl.success:
        raise RuntimeError(f"Erro na simulação não linear antes da perturbação: {sol_pre_nl.message}")

    X_pert_nl = sol_pre_nl.y[:, -1].copy()
    X_pert_nl[1] += delta_x

    sol_pos_nl = solve_ivp(
        fun=lambda tau, X: modelo_nao_linear(tau, X, delta_u),
        t_span=(t_pos_abs[0], t_pos_abs[-1]),
        y0=X_pert_nl,
        t_eval=t_pos_abs,
        rtol=1e-8,
        atol=1e-10,
        max_step=1e-5
    )

    if not sol_pos_nl.success:
        raise RuntimeError(f"Erro na simulação não linear após a perturbação: {sol_pos_nl.message}")

    t_plot = np.concatenate([t_pre_abs, t_pos_abs[1:]])
    x_nl = np.concatenate([sol_pre_nl.y[1], sol_pos_nl.y[1, 1:]])

    # Linear
    u_pre = np.zeros_like(tau_pre)
    _, y_pre_lin_delta, estados_pre_lin = signal.lsim(
        sistema_linear,
        U=u_pre,
        T=tau_pre,
        X0=np.array([0.0, 0.0, 0.0])
    )

    X_pert_lin = estados_pre_lin[-1, :].copy()
    X_pert_lin[1] += delta_x

    u_pos = np.zeros_like(tau_pos)
    _, y_pos_lin_delta, _ = signal.lsim(
        sistema_linear,
        U=u_pos,
        T=tau_pos,
        X0=X_pert_lin
    )

    y_pre_lin_delta = np.asarray(y_pre_lin_delta).reshape(-1)
    y_pos_lin_delta = np.asarray(y_pos_lin_delta).reshape(-1)

    x_lin = np.concatenate([
        x0 + y_pre_lin_delta,
        x0 + y_pos_lin_delta[1:]
    ])

    # Referência variável no tempo
    reference_signal = np.where(t_plot < t_pert, x0, x0 + delta_x)

    rmse = np.sqrt(np.mean((x_nl - x_lin) ** 2))

    caminho_png = salvar_grafico_png(
        t=t_plot,
        x_nl=x_nl,
        x_lin=x_lin,
        titulo=titulo,
        nome_base=nome_base,
        reference_signal=reference_signal,
        reference_label=reference_label,
        casas_decimais_y=3
    )

    print(f"{titulo}")
    print(f"  Perturbação aplicada em t = {t_pert:.4f} s")
    print(f"  Δx = {delta_x:.6e} m")
    print(f"  Δx = {delta_x * FATOR_POSICAO:.6e} cm")
    print(f"  RMSE = {rmse:.6e} m")
    print(f"  RMSE = {rmse * FATOR_POSICAO:.6e} cm")
    print(f"  PNG salvo em: {caminho_png}\n")


# ============================================================
# Execução
# ============================================================

def main():
    print("Validação local do modelo linearizado - v9\n")
    print(f"x0 = {x0:.6f} m = {x0 * FATOR_POSICAO:.6f} cm")
    print(f"i0 = {i0:.6f} A")
    print(f"u0 = {u0:.6f} V\n")
    print(f"Pasta de saída: {PASTA_SAIDA}\n")

    # Teste 1
    t1 = np.linspace(0, 0.20, 1000)
    simular(
        t=t1,
        delta_X0=np.array([0.0, 0.0, 0.0]),
        delta_u=lambda t: 0.0,
        titulo="Sem perturbação",
        nome_base="validacao_v9_teste1_sem_perturbacao",
        ylim=(0.9, 1.1),
        casas_decimais_y=3
    )

    # Teste 2
    delta_x_pert = 0.0001   # 0,1 mm = 0,01 cm
    t_pert_posicao = 0.02   # instante da perturbação [s]
    t2 = np.linspace(0, 0.10, 1000)

    simular_perturbacao_posicao_em_tempo(
        t=t2,
        t_pert=t_pert_posicao,
        delta_x=delta_x_pert,
        titulo="Perturbação na posição",
        nome_base="validacao_v9_teste2_perturbacao_posicao",
        reference_label="Referência de posição"
    )

    # Teste 3
    t3 = np.linspace(0, 0.15, 1000)
    t_degrau = 0.02
    amplitude_degrau = 0.05

    def degrau_tensao(t):
        return amplitude_degrau if t >= t_degrau else 0.0

    simular(
        t=t3,
        delta_X0=np.array([0.0, 0.0, 0.0]),
        delta_u=degrau_tensao,
        titulo="Degrau de tensão",
        nome_base="validacao_v9_teste3_degrau_tensao",
        event_time=t_degrau
    )


if __name__ == "__main__":
    main()
