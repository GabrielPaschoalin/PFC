#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador interativo OVERLAY com múltiplos eixos Y.

Formato:
    - HALL_INF_MV e HALL_SUP_MV no eixo Y esquerdo, em mV
    - PWM no eixo Y direito, em PWM
    - posição no segundo eixo Y direito, em mm
    - SYNC_PULSE fica disponível na legenda, oculto por padrão

Uso:
    python visualizar_mesclado_multieixo.py 2024_tratado_mesclado.txt

Saída:
    2024_tratado_mesclado_multieixo.html

Recursos:
    - tudo no mesmo gráfico
    - eixos Y independentes
    - zoom
    - pan
    - scroll zoom
    - legenda clicável
    - duplo clique na legenda para isolar curva
    - range slider no eixo X
"""

import argparse
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def carregar_tabela(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def encontrar_coluna(df: pd.DataFrame, candidatos):
    for c in candidatos:
        if c in df.columns:
            return c
    return None


def preparar_tempo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "T_US" in df.columns:
        t0 = pd.to_numeric(df["T_US"], errors="coerce").iloc[0]
        df["t_s"] = (pd.to_numeric(df["T_US"], errors="coerce") - t0) / 1_000_000.0
    elif "T_MS" in df.columns:
        t0 = pd.to_numeric(df["T_MS"], errors="coerce").iloc[0]
        df["t_s"] = (pd.to_numeric(df["T_MS"], errors="coerce") - t0) / 1000.0
    elif "time_sync_ms" in df.columns:
        t0 = pd.to_numeric(df["time_sync_ms"], errors="coerce").iloc[0]
        df["t_s"] = (pd.to_numeric(df["time_sync_ms"], errors="coerce") - t0) / 1000.0
    else:
        df["t_s"] = range(len(df))

    return df


def adicionar_trace(fig, df, coluna, nome, eixo_y, visivel=True):
    if coluna is None or coluna not in df.columns:
        return

    y = pd.to_numeric(df[coluna], errors="coerce")

    fig.add_trace(
        go.Scattergl(
            x=df["t_s"],
            y=y,
            mode="lines",
            name=nome,
            yaxis=eixo_y,
            visible=True if visivel else "legendonly",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "t = %{x:.6f} s<br>"
                "valor = %{y}<extra></extra>"
            ),
        )
    )


def gerar_figura(df: pd.DataFrame, titulo: str) -> go.Figure:
    df = preparar_tempo(df)

    col_pos = encontrar_coluna(df, ["posição", "posicao", "POSICAO", "topo_ruler_mm_median"])
    col_hall_inf = encontrar_coluna(df, ["HALL_INF_MV", "esp_HALL_INF_MV", "hall_inf_mv"])
    col_hall_sup = encontrar_coluna(df, ["HALL_SUP_MV", "esp_HALL_SUP_MV", "hall_sup_mv"])
    col_pwm = encontrar_coluna(df, ["PWM", "esp_PWM", "pwm", "pwm_atual"])
    col_sync = encontrar_coluna(df, ["SYNC_PULSE", "sync_pulse", "SYNC"])

    fig = go.Figure()

    # Eixo Y principal: Halls
    adicionar_trace(fig, df, col_hall_inf, "HALL_INF_MV [mV]", "y")
    adicionar_trace(fig, df, col_hall_sup, "HALL_SUP_MV [mV]", "y")

    # Eixo Y2: PWM
    adicionar_trace(fig, df, col_pwm, "PWM", "y2")

    # Eixo Y3: posição
    adicionar_trace(fig, df, col_pos, "posição [mm]", "y3")

    # SYNC opcional, oculto por padrão para não poluir.
    # Ele usa o eixo PWM porque é discreto 0/1.
    adicionar_trace(fig, df, col_sync, "SYNC_PULSE", "y2", visivel=False)

    fig.update_layout(
        title=titulo,
        height=780,
        hovermode="x unified",
        dragmode="zoom",

        # Deixa espaço à direita para dois eixos Y.
        xaxis=dict(
            title="tempo [s]",
            domain=[0.0, 0.84],
            rangeslider=dict(visible=True),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            showline=True,
            showgrid=True,
        ),

        # Eixo dos Halls, esquerdo.
        yaxis=dict(
            title="Halls [mV]",
            side="left",
            showgrid=True,
            zeroline=False,
            showspikes=True,
            spikesnap="cursor",
        ),

        # Eixo do PWM, direito mais interno.
        yaxis2=dict(
            title="PWM",
            overlaying="y",
            side="right",
            position=0.88,
            showgrid=False,
            zeroline=False,
            showspikes=True,
            spikesnap="cursor",
        ),

        # Eixo da posição, direito mais externo.
        yaxis3=dict(
            title="posição [mm]",
            overlaying="y",
            side="right",
            position=0.98,
            showgrid=False,
            zeroline=False,
            showspikes=True,
            spikesnap="cursor",
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),

        margin=dict(l=80, r=120, t=90, b=60),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualiza o arquivo mesclado em overlay com 3 eixos Y independentes."
    )
    parser.add_argument(
        "entrada",
        type=Path,
        help="Arquivo .txt ou .csv mesclado.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="Arquivo HTML de saída. Se omitido, usa entrada + _multieixo.html.",
    )
    parser.add_argument(
        "--nao-abrir",
        action="store_true",
        help="Não abre automaticamente no navegador.",
    )

    args = parser.parse_args()

    entrada = args.entrada
    if not entrada.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {entrada}")

    saida = args.saida
    if saida is None:
        saida = entrada.with_name(f"{entrada.stem}_multieixo.html")

    df = carregar_tabela(entrada)
    fig = gerar_figura(df, f"Overlay multieixo - {entrada.name}")

    fig.write_html(
        saida,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": entrada.stem,
                "height": 780,
                "width": 1600,
                "scale": 2,
            },
        },
    )

    print(f"HTML gerado: {saida}")

    if not args.nao_abrir:
        webbrowser.open(saida.resolve().as_uri())


if __name__ == "__main__":
    main()
