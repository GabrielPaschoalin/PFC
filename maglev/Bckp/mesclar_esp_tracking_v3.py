#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MESCLAR ESP + TRACKING V3

Alterações em relação ao mesclar_esp_tracking_v2.py:
- Usa o CSV tratado do vídeo como entrada principal de tracking.
  Formato esperado da v31:
    frame, frame_sync, tempo_ms, posicao_mm, posicao_raw_mm, valid, confidence, led_on
- Sincroniza usando múltiplos eventos:
    bordas de subida do PWM no CSV do ESP  <->  bordas de subida do LED no CSV tratado.
- Por padrão ignora a 1ª borda de subida do PWM e o 1º pulso longo do LED,
  começando o ajuste pela 2ª borda de subida do PWM.
- Remove automaticamente pulsos longos do LED, mantendo os pulsos curtos associados às bordas de subida do PWM.
- Ajusta uma relação linear de tempo:
    tempo_video_ms = offset + escala * tempo_esp_relativo_ms
- Mescla a posição interpolada do vídeo no CSV do ESP.
- Adiciona colunas de debug para facilitar diagnóstico caso o alinhamento dê errado.

Uso típico:
    python mesclar_esp_tracking_v3.py --esp 2024.csv --tracking 0105_tratado.csv --saida mesclado.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import bisect
import csv
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VERSION = "V3"


def parse_float(value) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().replace(',', '.')
    if text == '':
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_int(value) -> int:
    value_f = parse_float(value)
    if not math.isfinite(value_f):
        raise ValueError(f'Valor inteiro inválido: {value!r}')
    return int(value_f)


def truthy(value) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'sim', 'yes', 'y', 't', 'verdadeiro'}


def read_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f'Arquivo vazio: {path}')
    return header, rows


def require_column(rows: Sequence[Dict[str, str]], col: str, file_label: str) -> None:
    if not rows or col not in rows[0]:
        available = ', '.join(rows[0].keys()) if rows else '(sem colunas)'
        raise ValueError(f'O arquivo {file_label} precisa ter a coluna {col!r}. Colunas encontradas: {available}')


def first_existing_column(rows: Sequence[Dict[str, str]], candidates: Sequence[str], file_label: str) -> str:
    if not rows:
        raise ValueError(f'Arquivo {file_label} sem linhas.')
    for col in candidates:
        if col in rows[0]:
            return col
    available = ', '.join(rows[0].keys())
    raise ValueError(f'Não encontrei nenhuma das colunas {list(candidates)} no arquivo {file_label}. Colunas: {available}')


def format_float(value: float, digits: int = 9) -> str:
    if value is None or not math.isfinite(value):
        return ''
    return f'{value:.{digits}f}'.rstrip('0').rstrip('.')


def median(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    n = len(vals)
    if n == 0:
        return math.nan
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def fit_line(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, List[float], float, float]:
    """Retorna offset, escala, resíduos, mae e rmse para y = offset + escala*x."""
    if len(xs) != len(ys) or len(xs) == 0:
        raise ValueError('fit_line recebeu vetores incompatíveis.')
    if len(xs) == 1:
        offset = ys[0]
        scale = 1.0
        residuals = [0.0]
        return offset, scale, residuals, 0.0, 0.0

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if abs(var_x) < 1e-12:
        scale = 1.0
    else:
        cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        scale = cov_xy / var_x
    offset = y_mean - scale * x_mean
    residuals = [y - (offset + scale * x) for x, y in zip(xs, ys)]
    mae = sum(abs(r) for r in residuals) / len(residuals)
    rmse = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    return offset, scale, residuals, mae, rmse


def detect_rising_edges_from_numeric(
    rows: Sequence[Dict[str, str]],
    time_col: str,
    value_col: str,
    threshold: float = 0.5,
    min_interval_ms: float = 1.0,
) -> List[Dict[str, float]]:
    """Detecta bordas de subida quando value cruza threshold de baixo para cima."""
    require_column(rows, time_col, 'ESP')
    require_column(rows, value_col, 'ESP')
    edges: List[Dict[str, float]] = []
    prev_val = parse_float(rows[0].get(value_col))
    prev_high = math.isfinite(prev_val) and prev_val > threshold
    last_edge_ms = -math.inf

    for row in rows[1:]:
        cur_val = parse_float(row.get(value_col))
        if not math.isfinite(cur_val):
            cur_high = False
        else:
            cur_high = cur_val > threshold
        t_us = parse_int(row.get(time_col))
        t_ms = t_us / 1000.0
        if (not prev_high) and cur_high and (t_ms - last_edge_ms >= min_interval_ms):
            edges.append({
                'time_us': float(t_us),
                'time_ms': float(t_ms),
                'value': float(cur_val),
                'row_index': float(len(edges)),
            })
            last_edge_ms = t_ms
        prev_high = cur_high
    return edges


def detect_binary_pulses(
    rows: Sequence[Dict[str, str]],
    time_col: str,
    value_col: str,
) -> List[Dict[str, float]]:
    """Detecta pulsos booleanos e retorna rise, fall e largura.

    Se o arquivo começa com o sinal já alto, cria um pulso começando na primeira linha.
    Se termina alto, fecha o pulso na última linha.
    """
    require_column(rows, time_col, 'tracking')
    require_column(rows, value_col, 'tracking')

    pulses: List[Dict[str, float]] = []
    prev_on = truthy(rows[0].get(value_col))
    current_rise_ms: Optional[float] = parse_float(rows[0].get(time_col)) if prev_on else None
    current_rise_frame = parse_float(rows[0].get('frame')) if prev_on else math.nan

    for row in rows[1:]:
        cur_on = truthy(row.get(value_col))
        t = parse_float(row.get(time_col))
        frame = parse_float(row.get('frame'))
        if not math.isfinite(t):
            prev_on = cur_on
            continue
        if (not prev_on) and cur_on:
            current_rise_ms = t
            current_rise_frame = frame
        elif prev_on and (not cur_on) and current_rise_ms is not None:
            fall_ms = t
            pulses.append({
                'index_all': float(len(pulses)),
                'rise_ms': float(current_rise_ms),
                'fall_ms': float(fall_ms),
                'width_ms': float(fall_ms - current_rise_ms),
                'rise_frame': float(current_rise_frame),
                'fall_frame': float(frame),
            })
            current_rise_ms = None
            current_rise_frame = math.nan
        prev_on = cur_on

    if prev_on and current_rise_ms is not None:
        last = rows[-1]
        fall_ms = parse_float(last.get(time_col))
        fall_frame = parse_float(last.get('frame'))
        if math.isfinite(fall_ms):
            pulses.append({
                'index_all': float(len(pulses)),
                'rise_ms': float(current_rise_ms),
                'fall_ms': float(fall_ms),
                'width_ms': float(fall_ms - current_rise_ms),
                'rise_frame': float(current_rise_frame),
                'fall_frame': float(fall_frame),
            })
    return pulses


def select_led_pwm_pulses(
    led_pulses: Sequence[Dict[str, float]],
    expected_count: int,
    short_max_ms: float,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Seleciona os pulsos curtos do LED associados às bordas de subida do PWM."""
    warnings: List[str] = []
    if not led_pulses:
        raise ValueError('Não encontrei pulsos de LED no tracking tratado pela coluna led_on.')

    finite_widths = [p['width_ms'] for p in led_pulses if math.isfinite(p.get('width_ms', math.nan))]
    med_width = median(finite_widths)
    effective_short_max = short_max_ms
    if math.isfinite(med_width) and med_width < short_max_ms:
        # Mantém uma margem em relação aos pulsos curtos, sem deixar passar pulsos longos de 200 ms.
        effective_short_max = min(short_max_ms, max(2.8 * med_width, med_width + 12.0))

    short = [p for p in led_pulses if math.isfinite(p.get('width_ms', math.nan)) and p['width_ms'] <= effective_short_max]

    if expected_count > 0 and len(short) == expected_count:
        return list(short), warnings

    # Caso típico: LED tem pulso inicial longo + N pulsos curtos + pulso final longo.
    if expected_count > 0 and len(led_pulses) == expected_count + 2:
        warnings.append(
            'Quantidade de pulsos LED = PWM + 2. Removi primeiro e último pulso por serem sync inicial/final.'
        )
        return list(led_pulses[1:-1]), warnings

    # Se o filtro por largura encontrou mais pulsos que o esperado, usa os primeiros por ordem.
    if expected_count > 0 and len(short) > expected_count:
        warnings.append(
            f'Foram encontrados {len(short)} pulsos LED curtos para {expected_count} bordas PWM. Usei os primeiros {expected_count}.'
        )
        return list(short[:expected_count]), warnings

    # Se o filtro por largura encontrou menos pulsos, mas removendo os longos extremos fecha, usa esse fallback.
    if expected_count > 0 and len(led_pulses) >= expected_count:
        candidates = list(led_pulses)
        if len(candidates) >= 3:
            # Remove extremos longos quando forem muito maiores que a mediana dos demais.
            widths = [p.get('width_ms', math.nan) for p in candidates]
            sorted_widths = sorted(w for w in widths if math.isfinite(w))
            typical = median(sorted_widths[:max(1, len(sorted_widths) // 2)])
            if math.isfinite(typical):
                if candidates and candidates[0].get('width_ms', 0) > 3.0 * typical:
                    candidates = candidates[1:]
                    warnings.append('Removi primeiro pulso LED por largura alta em relação aos curtos.')
                if candidates and candidates[-1].get('width_ms', 0) > 3.0 * typical:
                    candidates = candidates[:-1]
                    warnings.append('Removi último pulso LED por largura alta em relação aos curtos.')
        if len(candidates) >= expected_count:
            warnings.append(
                f'Pareamento parcial/forçado: usei {expected_count} pulsos LED de {len(candidates)} candidatos.'
            )
            return list(candidates[:expected_count]), warnings

    # Último fallback: usa o mínimo disponível e avisa.
    n = min(expected_count, len(short)) if expected_count > 0 else len(short)
    if n <= 0:
        n = min(expected_count, len(led_pulses)) if expected_count > 0 else len(led_pulses)
        warnings.append('Nenhum pulso LED passou no filtro de largura curta. Usei pulsos LED por ordem como fallback.')
        return list(led_pulses[:n]), warnings

    warnings.append(
        f'Quantidade de pulsos não fechou: {expected_count} PWM rises vs {len(short)} LED curtos. Usei {n} pares.'
    )
    return list(short[:n]), warnings


def build_tracking_series(
    rows: Sequence[Dict[str, str]],
    time_col: str,
    pos_col: str,
    valid_col: Optional[str] = 'valid',
) -> Tuple[List[float], List[float]]:
    require_column(rows, time_col, 'tracking')
    require_column(rows, pos_col, 'tracking')
    pairs: List[Tuple[float, float]] = []
    for row in rows:
        if valid_col and valid_col in row and not truthy(row.get(valid_col)):
            continue
        t = parse_float(row.get(time_col))
        p = parse_float(row.get(pos_col))
        if math.isfinite(t) and math.isfinite(p):
            pairs.append((t, p))
    if len(pairs) < 2:
        raise ValueError('Poucos pontos válidos no tracking tratado para interpolar posição.')
    pairs.sort(key=lambda item: item[0])
    xs: List[float] = []
    ys: List[float] = []
    for t, p in pairs:
        if xs and abs(t - xs[-1]) < 1e-12:
            ys[-1] = p
        else:
            xs.append(t)
            ys.append(p)
    return xs, ys


def interpolate_linear_with_gap(
    x: float,
    xs: Sequence[float],
    ys: Sequence[float],
    max_gap_ms: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    if x < xs[0] or x > xs[-1]:
        return None, None
    idx = bisect.bisect_left(xs, x)
    if idx < len(xs) and abs(xs[idx] - x) < 1e-12:
        prev_gap = xs[idx] - xs[idx - 1] if idx > 0 else 0.0
        next_gap = xs[idx + 1] - xs[idx] if idx + 1 < len(xs) else 0.0
        return ys[idx], max(prev_gap, next_gap)
    if idx == 0 or idx >= len(xs):
        return None, None
    x0 = xs[idx - 1]
    x1 = xs[idx]
    y0 = ys[idx - 1]
    y1 = ys[idx]
    gap = x1 - x0
    if max_gap_ms is not None and math.isfinite(max_gap_ms) and gap > max_gap_ms:
        return None, gap
    if x1 == x0:
        return y0, gap
    alpha = (x - x0) / (x1 - x0)
    return y0 + alpha * (y1 - y0), gap


def find_esp_sync_edges(rows: Sequence[Dict[str, str]], time_col: str, sync_col: str) -> Tuple[Optional[float], Optional[float]]:
    if not rows or sync_col not in rows[0]:
        return None, None
    prev = parse_int(rows[0].get(sync_col))
    rise_us = None
    fall_us = None
    for row in rows[1:]:
        cur = parse_int(row.get(sync_col))
        t_us = parse_int(row.get(time_col))
        if rise_us is None and prev == 0 and cur == 1:
            rise_us = float(t_us)
        elif rise_us is not None and fall_us is None and prev == 1 and cur == 0:
            fall_us = float(t_us)
            break
        prev = cur
    return rise_us, fall_us


def merge_files(
    esp_path,
    tracking_path,
    output_path=None,
    nome_tratado=None,
    esp_time_col='T_US',
    esp_pwm_col='PWM',
    esp_sync_col='SYNC_PULSE',
    tracking_time_col='auto',
    tracking_pos_col='auto',
    tracking_led_col='led_on',
    pwm_rise_threshold=0.5,
    led_short_max_ms=80.0,
    sync_start_pwm_index=1,
    max_interp_gap_ms=100.0,
    add_debug_cols=True,
):
    esp_path = Path(esp_path)
    tracking_path = Path(tracking_path)
    esp_header, esp_rows = read_rows(esp_path)
    tracking_header, tracking_rows = read_rows(tracking_path)

    require_column(esp_rows, esp_time_col, 'ESP')
    require_column(esp_rows, esp_pwm_col, 'ESP')

    if tracking_time_col == 'auto':
        tracking_time_col = first_existing_column(tracking_rows, ['tempo_ms', 'time_sync_ms', 'time_ms'], 'tracking tratado')
    else:
        require_column(tracking_rows, tracking_time_col, 'tracking tratado')

    if tracking_pos_col == 'auto':
        tracking_pos_col = first_existing_column(
            tracking_rows,
            ['posicao_mm', 'posição', 'posicao', 'topo_ruler_mm_median', 'topo_ruler_mm'],
            'tracking tratado',
        )
    else:
        require_column(tracking_rows, tracking_pos_col, 'tracking tratado')

    require_column(tracking_rows, tracking_led_col, 'tracking tratado')

    esp_pwm_edges = detect_rising_edges_from_numeric(
        esp_rows,
        time_col=esp_time_col,
        value_col=esp_pwm_col,
        threshold=float(pwm_rise_threshold),
        min_interval_ms=1.0,
    )
    if not esp_pwm_edges:
        raise ValueError(f'Não encontrei bordas de subida de PWM na coluna {esp_pwm_col!r} do ESP.')

    led_pulses_all = detect_binary_pulses(tracking_rows, tracking_time_col, tracking_led_col)

    sync_start_pwm_index = int(sync_start_pwm_index)
    if sync_start_pwm_index < 0:
        raise ValueError('--sync-start-pwm-index não pode ser negativo.')
    if sync_start_pwm_index >= len(esp_pwm_edges):
        raise ValueError(
            f'--sync-start-pwm-index={sync_start_pwm_index} é inválido: só há {len(esp_pwm_edges)} bordas PWM.'
        )

    esp_pwm_edges_for_sync = list(esp_pwm_edges[sync_start_pwm_index:])
    if sync_start_pwm_index > 0:
        warnings = [
            f'Ignorei as {sync_start_pwm_index} primeiras bordas PWM no ajuste temporal. '
            f'O pareamento começa na borda PWM #{sync_start_pwm_index + 1}.'
        ]
    else:
        warnings = []

    led_pulses_selected, led_warnings = select_led_pwm_pulses(
        led_pulses_all,
        expected_count=len(esp_pwm_edges_for_sync),
        short_max_ms=float(led_short_max_ms),
    )
    warnings.extend(led_warnings)

    n_pairs = min(len(esp_pwm_edges_for_sync), len(led_pulses_selected))
    if n_pairs <= 0:
        raise ValueError('Não foi possível montar pares PWM rise <-> LED rise.')
    if n_pairs < len(esp_pwm_edges_for_sync):
        warnings.append(
            f'Usei apenas {n_pairs} pares de sincronismo para {len(esp_pwm_edges_for_sync)} bordas PWM usadas no ajuste.'
        )

    esp_ref_us = esp_pwm_edges_for_sync[0]['time_us']
    sync_x = [((esp_pwm_edges_for_sync[i]['time_us'] - esp_ref_us) / 1000.0) for i in range(n_pairs)]
    sync_y = [led_pulses_selected[i]['rise_ms'] for i in range(n_pairs)]
    offset_ms, scale, residuals, residual_mae, residual_rmse = fit_line(sync_x, sync_y)

    residual_by_us: Dict[int, Tuple[int, float]] = {}
    for i in range(n_pairs):
        residual_by_us[int(round(esp_pwm_edges_for_sync[i]['time_us']))] = (i, residuals[i])

    track_times, track_pos = build_tracking_series(
        tracking_rows,
        time_col=tracking_time_col,
        pos_col=tracking_pos_col,
        valid_col='valid' if 'valid' in tracking_rows[0] else None,
    )

    if output_path is None:
        if nome_tratado is None:
            nome_tratado = esp_path.stem
            if not nome_tratado.endswith('_tratado'):
                nome_tratado = f'{nome_tratado}_tratado'
        output_path = esp_path.with_name(f'{nome_tratado}_mesclado.csv')
    else:
        output_path = Path(output_path)

    output_header = list(esp_header)
    added_cols = ['posição']
    if add_debug_cols:
        added_cols += [
            'tempo_video_ms',
            'posicao_video_valida',
            'tracking_interp_gap_ms',
            'sync_pair_index',
            'sync_fit_residual_ms',
        ]
    for col in added_cols:
        if col not in output_header:
            output_header.append(col)

    written = 0
    skipped_before = 0
    skipped_after = 0
    pos_invalid = 0
    first_t_video = None
    last_t_video = None

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_header, extrasaction='ignore')
        writer.writeheader()
        for row in esp_rows:
            t_us = parse_int(row.get(esp_time_col))
            esp_rel_ms = (t_us - esp_ref_us) / 1000.0
            t_video_ms = offset_ms + scale * esp_rel_ms

            if t_video_ms < track_times[0]:
                skipped_before += 1
                continue
            if t_video_ms > track_times[-1]:
                skipped_after += 1
                continue

            pos, gap = interpolate_linear_with_gap(t_video_ms, track_times, track_pos, max_interp_gap_ms)
            out = dict(row)
            out['posição'] = format_float(pos) if pos is not None else ''
            if add_debug_cols:
                out['tempo_video_ms'] = format_float(t_video_ms)
                out['posicao_video_valida'] = '1' if pos is not None else '0'
                out['tracking_interp_gap_ms'] = format_float(gap) if gap is not None else ''
                pair = residual_by_us.get(int(round(t_us)))
                if pair is not None:
                    out['sync_pair_index'] = str(pair[0])
                    out['sync_fit_residual_ms'] = format_float(pair[1])
                else:
                    out['sync_pair_index'] = ''
                    out['sync_fit_residual_ms'] = ''
            writer.writerow(out)
            written += 1
            if pos is None:
                pos_invalid += 1
            if first_t_video is None:
                first_t_video = t_video_ms
            last_t_video = t_video_ms

    esp_sync_rise_us, esp_sync_fall_us = find_esp_sync_edges(esp_rows, esp_time_col, esp_sync_col)
    esp_sync_width_ms = None
    if esp_sync_rise_us is not None and esp_sync_fall_us is not None:
        esp_sync_width_ms = (esp_sync_fall_us - esp_sync_rise_us) / 1000.0

    selected_led_widths = [p['width_ms'] for p in led_pulses_selected]

    return {
        'versao': VERSION,
        'arquivo_saida': str(output_path),
        'linhas_esp_lidas': len(esp_rows),
        'linhas_tracking_lidas': len(tracking_rows),
        'linhas_mescladas': written,
        'linhas_sem_posicao_interpolada': pos_invalid,
        'linhas_cortadas_antes_tracking': skipped_before,
        'linhas_cortadas_apos_tracking': skipped_after,
        'coluna_tempo_tracking': tracking_time_col,
        'coluna_posicao_tracking': tracking_pos_col,
        'coluna_led_tracking': tracking_led_col,
        'bordas_pwm_esp_detectadas': len(esp_pwm_edges),
        'bordas_pwm_esp_ignoradas_inicio': sync_start_pwm_index,
        'bordas_pwm_esp_usadas_no_ajuste': len(esp_pwm_edges_for_sync),
        'pulsos_led_tracking_detectados_total': len(led_pulses_all),
        'pulsos_led_usados_para_pwm': len(led_pulses_selected),
        'pares_sincronismo_usados': n_pairs,
        'esp_primeiro_pwm_detectado_rise_us': int(round(esp_pwm_edges[0]['time_us'])),
        'esp_primeiro_pwm_usado_rise_us': int(round(esp_pwm_edges_for_sync[0]['time_us'])),
        'tracking_primeiro_led_usado_rise_ms': led_pulses_selected[0]['rise_ms'] if led_pulses_selected else None,
        'estrategia_sincronismo': f'PWM a partir do índice {sync_start_pwm_index} (1=segunda borda) pareado com pulsos LED curtos',
        'tempo_offset_ms': offset_ms,
        'fator_escala_tempo_esp_para_tracking': scale,
        'sync_residuo_mae_ms': residual_mae,
        'sync_residuo_rmse_ms': residual_rmse,
        'sync_residuo_max_abs_ms': max((abs(r) for r in residuals), default=math.nan),
        'led_width_ms_mediana_usados': median(selected_led_widths),
        'led_width_ms_min_usados': min(selected_led_widths) if selected_led_widths else math.nan,
        'led_width_ms_max_usados': max(selected_led_widths) if selected_led_widths else math.nan,
        'esp_sync_inicial_rise_us': esp_sync_rise_us,
        'esp_sync_inicial_fall_us': esp_sync_fall_us,
        'esp_sync_inicial_width_ms': esp_sync_width_ms,
        'primeiro_tempo_video_mesclado_ms': first_t_video,
        'ultimo_tempo_video_mesclado_ms': last_t_video,
        'tracking_inicio_ms': track_times[0],
        'tracking_fim_ms': track_times[-1],
        'avisos': ' | '.join(warnings) if warnings else '',
    }


def main():
    parser = argparse.ArgumentParser(description='Mescla CSV do ESP com tracking tratado do vídeo usando PWM rise <-> LED rise. V3 ignora por padrão a primeira borda PWM no ajuste.')
    parser.add_argument('--esp', default='2024.csv', help='CSV do ESP/Arduino com T_US e PWM.')
    parser.add_argument('--tracking', default='tracking_tratado.csv', help='CSV tratado do vídeo, ex.: 0105_tratado.csv.')
    parser.add_argument('--saida', default=None, help='Arquivo CSV de saída.')
    parser.add_argument('--nome-tratado', default=None, help='Nome base usado caso --saida não seja informado.')
    parser.add_argument('--esp-time-col', default='T_US')
    parser.add_argument('--esp-pwm-col', default='PWM')
    parser.add_argument('--esp-sync-col', default='SYNC_PULSE')
    parser.add_argument('--tracking-time-col', default='auto', help='auto, tempo_ms, time_sync_ms etc.')
    parser.add_argument('--tracking-pos-col', default='auto', help='auto, posicao_mm, topo_ruler_mm_median etc.')
    parser.add_argument('--tracking-led-col', default='led_on')
    parser.add_argument('--pwm-rise-threshold', type=float, default=0.5, help='Limiar para detectar PWM saindo de zero.')
    parser.add_argument('--led-short-max-ms', type=float, default=80.0, help='Largura máxima esperada para pulsos curtos do LED.')
    parser.add_argument(
        '--sync-start-pwm-index',
        type=int,
        default=1,
        help='Índice zero-based da primeira borda PWM usada no ajuste. Default=1: ignora a primeira borda e começa na segunda.',
    )
    parser.add_argument('--max-interp-gap-ms', type=float, default=100.0, help='Gap máximo permitido para interpolar posição.')
    parser.add_argument('--sem-debug-cols', action='store_true', help='Não adiciona colunas de debug na saída.')
    args = parser.parse_args()

    resumo = merge_files(
        esp_path=args.esp,
        tracking_path=args.tracking,
        output_path=args.saida,
        nome_tratado=args.nome_tratado,
        esp_time_col=args.esp_time_col,
        esp_pwm_col=args.esp_pwm_col,
        esp_sync_col=args.esp_sync_col,
        tracking_time_col=args.tracking_time_col,
        tracking_pos_col=args.tracking_pos_col,
        tracking_led_col=args.tracking_led_col,
        pwm_rise_threshold=args.pwm_rise_threshold,
        led_short_max_ms=args.led_short_max_ms,
        sync_start_pwm_index=args.sync_start_pwm_index,
        max_interp_gap_ms=args.max_interp_gap_ms,
        add_debug_cols=not args.sem_debug_cols,
    )

    for k, v in resumo.items():
        if isinstance(v, float):
            print(f'{k}: {format_float(v)}')
        else:
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
