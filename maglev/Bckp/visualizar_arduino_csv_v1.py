#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUALIZAR ARDUINO CSV V1

Objetivo:
    Gerar um HTML interativo a partir do CSV bruto obtido no Arduino IDE/ESP.

Alterações/recursos da V1:
    - Pensado para arquivos como 0105.csv, com colunas T_US, HALL_INF_MV,
      HALL_SUP_MV, PWM e SYNC_PULSE.
    - Usa o CSV bruto, sem precisar do arquivo mesclado com vídeo.
    - Plota sensores analógicos no eixo Y esquerdo.
    - Plota PWM no eixo Y direito.
    - Plota SYNC_PULSE escalado no eixo do PWM, oculto por padrão.
    - Adiciona marcadores das bordas de subida do PWM e do SYNC_PULSE,
      ocultos por padrão para não poluir.
    - Gera HTML com zoom, pan, range slider, legenda clicável e hover unificado.

Uso:
    python visualizar_arduino_csv_v1.py 0105.csv

Saída padrão:
    0105_arduino_visual.html
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.graph_objects as go


TIME_CANDIDATES = ["T_US", "T_MS", "tempo_ms", "time_ms", "t_ms", "tempo_s", "time_s"]
PWM_CANDIDATES = ["PWM", "pwm", "pwm_atual", "PWM_APLICADO"]
SYNC_CANDIDATES = ["SYNC_PULSE", "sync_pulse", "SYNC", "LED", "led_on"]
SENSOR_PREFERRED = [
    "HALL_INF_MV",
    "HALL_SUP_MV",
    "HALL_INF_RAW",
    "HALL_SUP_RAW",
    "SENSOR_INF",
    "SENSOR_SUP",
]


def carregar_tabela(caminho: Path) -> pd.DataFrame:
    """Carrega CSV com separador inferido e limpa nomes de colunas."""
    df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError(f"Arquivo vazio: {caminho}")
    return df


def encontrar_coluna(df: pd.DataFrame, candidatos: Iterable[str]) -> Optional[str]:
    for c in candidatos:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidatos:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def para_numero(serie: pd.Series) -> pd.Series:
    """Converte número aceitando decimal com vírgula."""
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")
    return pd.to_numeric(
        serie.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def preparar_tempo(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Cria coluna t_s relativa ao início do arquivo."""
    df = df.copy()
    time_col = encontrar_coluna(df, TIME_CANDIDATES)

    if time_col is None:
        df["t_s"] = range(len(df))
        return df, "amostra"

    t = para_numero(df[time_col])
    t0 = t.dropna().iloc[0] if t.notna().any() else 0.0

    name = time_col.upper()
    if name.endswith("_US") or name == "T_US":
        df["t_s"] = (t - t0) / 1_000_000.0
    elif name.endswith("_MS") or name in {"T_MS", "TEMPO_MS", "TIME_MS"}:
        df["t_s"] = (t - t0) / 1000.0
    elif name.endswith("_S") or name in {"TEMPO_S", "TIME_S"}:
        df["t_s"] = t - t0
    else:
        # Fallback conservador: assume segundos se o nome não indica unidade.
        df["t_s"] = t - t0

    return df, time_col


def escolher_sensores(df: pd.DataFrame, pwm_col: Optional[str], sync_col: Optional[str], time_col: Optional[str]) -> list[str]:
    """Escolhe colunas de sensores para plotar no eixo esquerdo."""
    ignorar = {"t_s"}
    for c in [pwm_col, sync_col, time_col]:
        if c:
            ignorar.add(c)

    sensores = []

    for c in SENSOR_PREFERRED:
        real = encontrar_coluna(df, [c])
        if real and real not in ignorar and real not in sensores:
            sensores.append(real)

    for c in df.columns:
        if c in ignorar or c in sensores:
            continue
        y = para_numero(df[c])
        # Considera sensor se a maior parte da coluna for numérica.
        if y.notna().mean() >= 0.75:
            sensores.append(c)

    return sensores


def bordas_subida(y: pd.Series) -> pd.Series:
    """Retorna máscara booleana de borda de subida para sinais discretos."""
    vals = para_numero(y).fillna(0)
    prev = vals.shift(1).fillna(vals.iloc[0])
    return (prev <= 0) & (vals > 0)


def adicionar_trace(fig: go.Figure, x, y, nome: str, eixo_y: str, visivel=True, modo="lines"):
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode=modo,
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
    df, time_col = preparar_tempo(df)

    pwm_col = encontrar_coluna(df, PWM_CANDIDATES)
    sync_col = encontrar_coluna(df, SYNC_CANDIDATES)
    sensores = escolher_sensores(df, pwm_col, sync_col, time_col)

    fig = go.Figure()

    for col in sensores:
        adicionar_trace(fig, df["t_s"], para_numero(df[col]), col, "y")

    pwm_max = 255.0
    if pwm_col:
        pwm = para_numero(df[pwm_col])
        if pwm.notna().any():
            pwm_max = max(1.0, float(pwm.max()))
        adicionar_trace(fig, df["t_s"], pwm, "PWM", "y2")

        rise = bordas_subida(df[pwm_col])
        if rise.any():
            fig.add_trace(
                go.Scattergl(
                    x=df.loc[rise, "t_s"],
                    y=[pwm_max] * int(rise.sum()),
                    mode="markers",
                    name="PWM rising edge",
                    yaxis="y2",
                    visible="legendonly",
                    marker=dict(symbol="triangle-up", size=9),
                    hovertemplate="<b>PWM rising edge</b><br>t = %{x:.6f} s<extra></extra>",
                )
            )

    if sync_col:
        sync = para_numero(df[sync_col]).fillna(0)
        sync_scaled = sync * pwm_max
        adicionar_trace(fig, df["t_s"], sync_scaled, f"{sync_col} x {pwm_max:g}", "y2", visivel=False)

        sync_rise = bordas_subida(df[sync_col])
        if sync_rise.any():
            fig.add_trace(
                go.Scattergl(
                    x=df.loc[sync_rise, "t_s"],
                    y=[pwm_max] * int(sync_rise.sum()),
                    mode="markers",
                    name=f"{sync_col} rising edge",
                    yaxis="y2",
                    visible="legendonly",
                    marker=dict(symbol="circle", size=8),
                    hovertemplate=f"<b>{sync_col} rising edge</b><br>t = %{{x:.6f}} s<extra></extra>",
                )
            )

    if not sensores and not pwm_col and not sync_col:
        raise ValueError("Não encontrei colunas numéricas úteis para plotar.")

    fig.update_layout(
        title=titulo,
        height=820,
        hovermode="x unified",
        dragmode="zoom",
        xaxis=dict(
            title="tempo [s]",
            domain=[0.0, 0.86],
            rangeslider=dict(visible=True),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            showline=True,
            showgrid=True,
        ),
        yaxis=dict(
            title="sensores",
            side="left",
            showgrid=True,
            zeroline=False,
            showspikes=True,
            spikesnap="cursor",
        ),
        yaxis2=dict(
            title="PWM / sync escalado",
            overlaying="y",
            side="right",
            position=0.98,
            range=[-5, max(260.0, pwm_max * 1.08)],
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
        margin=dict(l=80, r=110, t=95, b=60),
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=-0.18,
        showarrow=False,
        align="left",
        text=(
            f"Coluna de tempo: {time_col or 'índice de amostra'} | "
            f"Sensores: {', '.join(sensores) if sensores else 'nenhum'} | "
            f"PWM: {pwm_col or 'não encontrado'} | "
            f"Sync: {sync_col or 'não encontrado'}"
        ),
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera HTML interativo para visualizar CSV bruto do Arduino/ESP."
    )
    parser.add_argument(
        "entrada",
        type=Path,
        help="Arquivo .csv obtido no Arduino IDE/Serial Monitor.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="Arquivo HTML de saída. Se omitido, usa entrada + _arduino_visual.html.",
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

    saida = args.saida or entrada.with_name(f"{entrada.stem}_arduino_visual.html")

    df = carregar_tabela(entrada)
    fig = gerar_figura(df, f"Visualização Arduino/ESP - {entrada.name}")

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
                "height": 820,
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
