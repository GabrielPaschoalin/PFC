#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_esp_key_value_csv(path: str | Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d: dict[str, str] = {}
            for part in line.split(','):
                if ':' not in part:
                    continue
                k, v = part.split(':', 1)
                d[k.strip()] = v.strip()
            if not d:
                continue
            if 'T_US' in d:
                rows.append(d)
            else:
                summaries.append(d)

    if not rows:
        raise ValueError(f"Nenhuma linha com T_US encontrada no arquivo ESP: {path}")

    df = pd.DataFrame(rows)
    df = df.replace({"NULL": np.nan, "null": np.nan, "NaN": np.nan, "nan": np.nan, "": np.nan})
    try:
        df = df.infer_objects(copy=False)
    except TypeError:
        df = df.infer_objects()

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0 and converted.notna().sum() >= df[col].notna().sum() * 0.80:
            df[col] = converted

    df['T_US'] = pd.to_numeric(df['T_US'], errors='coerce')
    df = df.dropna(subset=['T_US']).sort_values('T_US').drop_duplicates(subset=['T_US'], keep='last')

    summary: dict[str, str] = {}
    for s in summaries:
        summary.update(s)
    return df.reset_index(drop=True), summary


def find_sync_rise_time_esp_us(df: pd.DataFrame) -> float:
    if 'SYNC_PULSE' not in df.columns:
        raise ValueError('ESP não possui SYNC_PULSE para sincronizar.')
    s = pd.to_numeric(df['SYNC_PULSE'], errors='coerce').fillna(0).astype(int)
    rise = df.index[(s.diff().fillna(s) > 0) & (s == 1)]
    if len(rise) > 0:
        return float(df.loc[int(rise[0]), 'T_US'])
    high = df.index[s == 1]
    if len(high) > 0:
        return float(df.loc[int(high[0]), 'T_US'])
    raise ValueError('Não encontrei borda de subida nem nível alto em SYNC_PULSE no ESP.')


def prepare_camera(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'time_sync_ms' in df.columns:
        df['TIME_SYNC_MS'] = pd.to_numeric(df['time_sync_ms'], errors='coerce')
    elif 'time_real_ms' in df.columns and 'sync_rising_edge' in df.columns:
        t = pd.to_numeric(df['time_real_ms'], errors='coerce')
        r = pd.to_numeric(df['sync_rising_edge'], errors='coerce').fillna(0).astype(int)
        idx = df.index[r == 1]
        if len(idx) == 0:
            raise ValueError('Câmera não tem time_sync_ms e não encontrei sync_rising_edge.')
        df['TIME_SYNC_MS'] = t - float(t.loc[int(idx[0])])
    else:
        raise ValueError('Câmera precisa de time_sync_ms ou de time_real_ms + sync_rising_edge.')

    df = df.dropna(subset=['TIME_SYNC_MS']).sort_values('TIME_SYNC_MS')
    df = df.drop_duplicates(subset=['TIME_SYNC_MS'], keep='last')
    return df.reset_index(drop=True)


def prepare_esp(path: str | Path, sync_anchor: str = 'rising') -> pd.DataFrame:
    df, summary = parse_esp_key_value_csv(path)
    t0 = find_sync_rise_time_esp_us(df)
    if sync_anchor.lower() != 'rising':
        raise ValueError('Por enquanto este script usa a borda de subida como âncora robusta.')
    df['TIME_SYNC_MS'] = (pd.to_numeric(df['T_US'], errors='coerce') - t0) / 1000.0
    df['ESP_SYNC_T0_US'] = t0
    for k, v in summary.items():
        df[f'SUMMARY_{k}'] = v
    return df.sort_values('TIME_SYNC_MS').drop_duplicates(subset=['TIME_SYNC_MS'], keep='last').reset_index(drop=True)


def is_hold_column(col: str) -> bool:
    u = col.upper()
    tokens = [
        'FRAME', 'VALID', 'STATE', 'TYPE', 'MOTIVO', 'TEST_ID', 'TEST_PARAM', 'SYNC', 'PWM',
        'REP', 'FASE', 'RANGE_STATUS', 'NEW_TOF', 'TOF_VALID', 'ACTIVE', 'METHOD', 'REASON',
        'ROI', 'CONFIG', 'SOURCE', 'CONFIDENCE', 'COUNT', 'PULSE', 'PULSO', 'LIM_', 'SUMMARY',
        'T_US', 'T0_US', 'FALLING', 'RISING', 'EVENT', 'OK'
    ]
    return any(tok in u for tok in tokens)


def resample_to_grid(src: pd.DataFrame, grid: pd.DataFrame, prefix: str) -> pd.DataFrame:
    src = src.sort_values('TIME_SYNC_MS').drop_duplicates(subset=['TIME_SYNC_MS'], keep='last').reset_index(drop=True)
    times = src['TIME_SYNC_MS'].astype(float).to_numpy()
    base = grid[['TIME_SYNC_MS']].copy()
    out = base.copy()

    if len(times) == 0:
        return out

    base_t = base['TIME_SYNC_MS'].to_numpy(dtype=float)

    for col in src.columns:
        if col == 'TIME_SYNC_MS':
            continue
        name = f'{prefix}{col}'
        series = src[col]
        numeric = pd.api.types.is_numeric_dtype(series)

        if numeric and not is_hold_column(col):
            y = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
            ok = np.isfinite(times) & np.isfinite(y)
            if ok.sum() >= 2:
                vals = np.interp(base_t, times[ok], y[ok])
                vals[base_t < np.nanmin(times[ok])] = np.nan
                vals[base_t > np.nanmax(times[ok])] = np.nan
                out[name] = vals
            elif ok.sum() == 1:
                vals = np.full(len(base), np.nan)
                vals[np.isclose(base_t, times[ok][0])] = y[ok][0]
                out[name] = vals
            else:
                out[name] = np.nan
        else:
            tmp = src[['TIME_SYNC_MS', col]].copy().sort_values('TIME_SYNC_MS')
            merged = pd.merge_asof(base.sort_values('TIME_SYNC_MS'), tmp, on='TIME_SYNC_MS', direction='backward')
            out[name] = merged[col].to_numpy()

    exact_times = pd.Series(times).round(6).astype(str)
    grid_times = base['TIME_SYNC_MS'].round(6).astype(str)
    out[f'{prefix}HAS_EXACT_SAMPLE'] = grid_times.isin(set(exact_times)).to_numpy()
    return out


def build_grid(cam: pd.DataFrame, esp: pd.DataFrame, mode: str, dt_ms: float | None) -> pd.DataFrame:
    mode = mode.lower()
    if mode == 'camera':
        t = cam['TIME_SYNC_MS'].astype(float).to_numpy()
    elif mode == 'esp':
        t = esp['TIME_SYNC_MS'].astype(float).to_numpy()
    elif mode == 'union':
        t = np.r_[cam['TIME_SYNC_MS'].astype(float).to_numpy(), esp['TIME_SYNC_MS'].astype(float).to_numpy()]
    elif mode == 'uniform':
        if dt_ms is None or dt_ms <= 0:
            raise ValueError('Para --grid uniform, informe --dt-ms positivo.')
        start = max(float(cam['TIME_SYNC_MS'].min()), float(esp['TIME_SYNC_MS'].min()))
        end = min(float(cam['TIME_SYNC_MS'].max()), float(esp['TIME_SYNC_MS'].max()))
        t = np.arange(start, end + 0.5 * dt_ms, dt_ms)
    else:
        raise ValueError('--grid deve ser camera, esp, union ou uniform.')

    t = np.array(sorted(set(np.round(t[np.isfinite(t)], 6))))
    return pd.DataFrame({'TIME_SYNC_MS': t})


def choose_ball_height_column(df: pd.DataFrame) -> str | None:
    candidates = [
        'cam_topo_ruler_mm_median',
        'cam_topo_ruler_mm',
        'cam_y_top_subpx_median',
        'cam_y_center_px_median',
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_lines(df: pd.DataFrame, time_col: str, y_cols: list[str], title: str, ylabel: str, out_path: Path) -> None:
    valid_cols = [c for c in y_cols if c in df.columns]
    if not valid_cols:
        return

    t = pd.to_numeric(df[time_col], errors='coerce')
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    plotted_any = False
    for c in valid_cols:
        y = pd.to_numeric(df[c], errors='coerce')
        if y.notna().sum() == 0:
            continue
        ax.plot(t, y, label=c)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel('TIME_SYNC_MS (ms)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(valid_cols) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pwm_with_height(df: pd.DataFrame, out_path: Path, height_col: str | None) -> None:
    if 'esp_PWM' not in df.columns and not height_col:
        return

    t = pd.to_numeric(df['TIME_SYNC_MS'], errors='coerce')
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111)
    drew = False

    if 'esp_PWM' in df.columns:
        y1 = pd.to_numeric(df['esp_PWM'], errors='coerce')
        if y1.notna().sum() > 0:
            ax1.step(t, y1, where='post', label='esp_PWM')
            ax1.set_ylabel('PWM')
            drew = True

    ax2 = None
    if height_col and height_col in df.columns:
        y2 = pd.to_numeric(df[height_col], errors='coerce')
        if y2.notna().sum() > 0:
            ax2 = ax1.twinx()
            ax2.plot(t, y2, label=height_col)
            ax2.set_ylabel('Altura da bolinha (mm)')
            drew = True

    if not drew:
        plt.close(fig)
        return

    ax1.set_title('PWM e altura da bolinha ao longo do tempo')
    ax1.set_xlabel('TIME_SYNC_MS (ms)')
    ax1.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    if handles:
        ax1.legend(handles, labels, loc='best')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_plots(merged: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    ball_height_col = choose_ball_height_column(merged)
    plot_lines(merged, 'TIME_SYNC_MS', ['esp_PWM'], 'PWM ao longo do tempo', 'PWM', plots_dir / '01_esp_pwm.png')
    plot_lines(merged, 'TIME_SYNC_MS', ['esp_HALL_INF_MV', 'esp_HALL_INF_FILT_MV'],
               'Hall inferior ao longo do tempo', 'mV', plots_dir / '02_hall_inferior.png')
    plot_lines(merged, 'TIME_SYNC_MS', ['esp_HALL_SUP_MV', 'esp_HALL_SUP_FILT_MV'],
               'Hall superior ao longo do tempo', 'mV', plots_dir / '03_hall_superior.png')
    if ball_height_col:
        plot_lines(merged, 'TIME_SYNC_MS', [ball_height_col],
                   f'Altura da bolinha ao longo do tempo ({ball_height_col})',
                   'mm', plots_dir / '04_altura_bolinha.png')
    plot_pwm_with_height(merged, plots_dir / '05_pwm_e_altura.png', ball_height_col)



def main() -> None:
    ap = argparse.ArgumentParser(description='Mescla CSV da câmera e log CSV key:value do ESP em um único TIME_SYNC_MS e gera gráficos.')
    ap.add_argument('--camera', required=True, help='CSV gerado pelo tracker da câmera, ex: tracking_tratado.csv')
    ap.add_argument('--esp', required=True, help='CSV/log do ESP, ex: ident_1610.csv')
    ap.add_argument('--out', required=True, help='CSV de saída sincronizado')
    ap.add_argument('--grid', choices=['camera', 'esp', 'union', 'uniform'], default='camera',
                    help='Base temporal do CSV final. Recomendo camera para comparar posição vídeo vs Hall/PWM.')
    ap.add_argument('--dt-ms', type=float, default=None, help='Passo em ms quando --grid uniform')
    ap.add_argument('--trim-overlap', action='store_true', help='Mantém apenas intervalo comum aos dois arquivos')
    ap.add_argument('--plots-dir', default=None, help='Pasta para salvar gráficos. Default: <pasta do out>/graficos')
    args = ap.parse_args()

    cam = prepare_camera(args.camera)
    esp = prepare_esp(args.esp)
    grid = build_grid(cam, esp, args.grid, args.dt_ms)

    if args.trim_overlap:
        start = max(float(cam['TIME_SYNC_MS'].min()), float(esp['TIME_SYNC_MS'].min()))
        end = min(float(cam['TIME_SYNC_MS'].max()), float(esp['TIME_SYNC_MS'].max()))
        grid = grid[(grid['TIME_SYNC_MS'] >= start) & (grid['TIME_SYNC_MS'] <= end)].reset_index(drop=True)

    cam_resampled = resample_to_grid(cam, grid, 'cam_')
    esp_resampled = resample_to_grid(esp, grid, 'esp_')

    merged = grid.merge(cam_resampled, on='TIME_SYNC_MS', how='left').merge(esp_resampled, on='TIME_SYNC_MS', how='left')

    priority = [
        'TIME_SYNC_MS',
        'cam_frame', 'cam_topo_ruler_mm', 'cam_topo_ruler_mm_median', 'cam_valid', 'cam_sync_pulse',
        'esp_T_US', 'esp_PWM', 'esp_HALL_INF_MV', 'esp_HALL_INF_FILT_MV', 'esp_HALL_SUP_MV', 'esp_HALL_SUP_FILT_MV',
        'esp_SYNC_PULSE', 'esp_IDENT_STATE', 'esp_IDENT_REP', 'esp_IDENT_FASE',
        'cam_HAS_EXACT_SAMPLE', 'esp_HAS_EXACT_SAMPLE'
    ]
    cols = [c for c in priority if c in merged.columns] + [c for c in merged.columns if c not in priority]
    merged = merged[cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    plots_dir = Path(args.plots_dir) if args.plots_dir else out_path.parent / 'graficos'
    generate_plots(merged, plots_dir)

    print(f"Camera: {len(cam)} linhas | TIME_SYNC {cam['TIME_SYNC_MS'].min():.3f}..{cam['TIME_SYNC_MS'].max():.3f} ms")
    print(f"ESP:    {len(esp)} linhas | TIME_SYNC {esp['TIME_SYNC_MS'].min():.3f}..{esp['TIME_SYNC_MS'].max():.3f} ms")
    print(f"Saída:  {len(merged)} linhas -> {out_path}")
    print(f"Gráficos salvos em: {plots_dir}")


if __name__ == '__main__':
    main()
