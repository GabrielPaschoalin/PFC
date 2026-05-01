from pathlib import Path
import csv
import math
import argparse
import bisect


def parse_float(value):
    if value is None:
        return math.nan
    text = str(value).strip().replace(',', '.')
    if text == '':
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_int(value):
    return int(float(str(value).strip().replace(',', '.')))


def truthy(value):
    return str(value).strip().lower() in {'1', 'true', 'sim', 'yes', 'y'}


def read_rows(path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f'Arquivo vazio: {path}')
    return header, rows


def find_esp_edges(rows, time_col='T_US', sync_col='SYNC_PULSE'):
    if time_col not in rows[0] or sync_col not in rows[0]:
        raise ValueError(f'O arquivo ESP precisa ter as colunas {time_col} e {sync_col}.')
    prev = parse_int(rows[0][sync_col])
    rise_us = None
    fall_us = None
    for row in rows[1:]:
        cur = parse_int(row[sync_col])
        t_us = parse_int(row[time_col])
        if rise_us is None and prev == 0 and cur == 1:
            rise_us = t_us
        elif rise_us is not None and fall_us is None and prev == 1 and cur == 0:
            fall_us = t_us
            break
        prev = cur
    if rise_us is None or fall_us is None:
        raise ValueError('Não encontrei rising edge e falling edge no SYNC_PULSE do ESP.')
    if fall_us <= rise_us:
        raise ValueError('Falling edge do ESP veio antes do rising edge.')
    return rise_us, fall_us


def find_tracking_edges(rows, time_col='time_sync_ms'):
    if time_col not in rows[0]:
        raise ValueError(f'O tracking precisa ter a coluna {time_col}.')
    rise_ms = None
    fall_ms = None
    for row in rows:
        event = str(row.get('sync_event', '')).strip().upper()
        if rise_ms is None and (truthy(row.get('sync_rising_edge', '0')) or event == 'RISING_EDGE'):
            rise_ms = parse_float(row[time_col])
        if fall_ms is None and (truthy(row.get('sync_falling_edge', '0')) or event == 'FALLING_EDGE'):
            fall_ms = parse_float(row[time_col])
        if rise_ms is not None and fall_ms is not None:
            break
    if rise_ms is None or fall_ms is None:
        raise ValueError('Não encontrei rising edge e falling edge no tracking tratado.')
    if fall_ms <= rise_ms:
        raise ValueError('Falling edge do tracking veio antes do rising edge.')
    return rise_ms, fall_ms


def build_tracking_series(rows, time_col='time_sync_ms', pos_col='topo_ruler_mm_median'):
    if pos_col not in rows[0]:
        raise ValueError(f'O tracking precisa ter a coluna {pos_col}.')
    pairs = []
    for row in rows:
        t = parse_float(row.get(time_col))
        p = parse_float(row.get(pos_col))
        if math.isfinite(t) and math.isfinite(p):
            pairs.append((t, p))
    if len(pairs) < 2:
        raise ValueError('Poucos pontos válidos no tracking para interpolar.')
    pairs.sort(key=lambda item: item[0])
    xs = []
    ys = []
    for t, p in pairs:
        if xs and abs(t - xs[-1]) < 1e-12:
            ys[-1] = p
        else:
            xs.append(t)
            ys.append(p)
    return xs, ys


def interpolate_linear(x, xs, ys):
    if x < xs[0] or x > xs[-1]:
        return None
    idx = bisect.bisect_left(xs, x)
    if idx < len(xs) and abs(xs[idx] - x) < 1e-12:
        return ys[idx]
    if idx == 0 or idx >= len(xs):
        return None
    x0 = xs[idx - 1]
    x1 = xs[idx]
    y0 = ys[idx - 1]
    y1 = ys[idx]
    if x1 == x0:
        return y0
    alpha = (x - x0) / (x1 - x0)
    return y0 + alpha * (y1 - y0)


def format_float(value):
    return f'{value:.9f}'.rstrip('0').rstrip('.')


def merge_files(esp_path, tracking_path, output_path=None, nome_tratado=None):
    esp_path = Path(esp_path)
    tracking_path = Path(tracking_path)
    esp_header, esp_rows = read_rows(esp_path)
    _, tracking_rows = read_rows(tracking_path)

    esp_rise_us, esp_fall_us = find_esp_edges(esp_rows)
    track_rise_ms, track_fall_ms = find_tracking_edges(tracking_rows)
    track_times, track_pos = build_tracking_series(tracking_rows)

    esp_pulse_ms = (esp_fall_us - esp_rise_us) / 1000.0
    track_pulse_ms = track_fall_ms - track_rise_ms

    if output_path is None:
        if nome_tratado is None:
            nome_tratado = esp_path.stem
            if not nome_tratado.endswith('_tratado'):
                nome_tratado = f'{nome_tratado}_tratado'
        output_path = esp_path.with_name(f'{nome_tratado}_mesclado.csv')
    else:
        output_path = Path(output_path)

    output_header = list(esp_header)
    if 'posição' not in output_header:
        output_header.append('posição')

    written = 0
    skipped_before = 0
    skipped_after = 0
    first_t_aligned = None
    last_t_aligned = None
    scale = 1.0
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_header, extrasaction='ignore')
        writer.writeheader()
        for row in esp_rows:
            t_us = parse_int(row['T_US'])
            t_aligned_ms = track_rise_ms + ((t_us - esp_rise_us) / 1000.0)
            if t_aligned_ms < track_times[0]:
                skipped_before += 1
                continue
            if t_aligned_ms > track_times[-1]:
                skipped_after += 1
                continue
            pos = interpolate_linear(t_aligned_ms, track_times, track_pos)
            if pos is None:
                skipped_after += 1
                continue
            out = dict(row)
            out['posição'] = format_float(pos)
            writer.writerow(out)
            written += 1
            if first_t_aligned is None:
                first_t_aligned = t_aligned_ms
            last_t_aligned = t_aligned_ms

    return {
        'arquivo_saida': str(output_path),
        'linhas_esp_lidas': len(esp_rows),
        'linhas_tracking_lidas': len(tracking_rows),
        'linhas_mescladas': written,
        'linhas_cortadas_antes_sync': skipped_before,
        'linhas_cortadas_apos_fim_tracking': skipped_after,
        'esp_rising_us': esp_rise_us,
        'esp_falling_us': esp_fall_us,
        'esp_largura_pulso_ms': esp_pulse_ms,
        'tracking_rising_ms': track_rise_ms,
        'tracking_falling_ms': track_fall_ms,
        'tracking_largura_pulso_ms': track_pulse_ms,
        'fator_escala_tempo_esp_para_tracking': scale,
        'primeiro_tempo_alinhado_ms': first_t_aligned,
        'ultimo_tempo_alinhado_ms': last_t_aligned,
        'tracking_inicio_ms': track_times[0],
        'tracking_fim_ms': track_times[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esp', default='2024.csv')
    parser.add_argument('--tracking', default='tracking_tratado.csv')
    parser.add_argument('--saida', default=None)
    parser.add_argument('--nome-tratado', default=None)
    args = parser.parse_args()
    resumo = merge_files(args.esp, args.tracking, args.saida, args.nome_tratado)
    for k, v in resumo.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
