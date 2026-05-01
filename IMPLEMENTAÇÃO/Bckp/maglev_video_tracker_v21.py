#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAGLEV VIDEO TRACKER V21 - SIMPLE DARK-BLOB TOP TRACKER + REJECTION LOGS

Objetivo:
    Rastrear diretamente o TOPO da bolinha preta/escura e converter para mm usando
    uma regua reta, fixa, vertical, com zero embaixo e altura conhecida de 26 mm.

Mudanca principal da V16:
    - Remove a dependencia de Hough/circulos como medida principal.
    - Remove a calibracao inclinada/complexa da regua.
    - Usa a bolinha como o maior blob escuro plausivel no frame inteiro.
    - Mede o topo diretamente pelo contorno/marcara da bolinha.
    - Converte diretamente:
          topo_ruler_mm = (ruler_zero_y_px - y_top_ball_px) / px_per_mm

Saidas:
    tracking_full.csv
    tracking_tratado.csv
    candidates.csv
    preview.jpg
    plot.png
    ruler_preview.jpg
    sync_preview.jpg
    mask_preview.jpg
    rejection_log.csv
    tracking_decision_log.csv
    tracking_errors.txt

Dependencias:
    pip install opencv-python numpy pandas matplotlib

Exemplo:
    python maglev_video_tracker_v16.py --video video.mp4 --output-dir saida_v16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Parametros
# ============================================================

@dataclass
class Params:
    # Regua fisica: reta, fixa, zero embaixo.
    ruler_height_mm: float = 26.0
    ruler_search_x0_ratio: float = 0.50       # procurar regua na metade direita do frame
    ruler_min_height_ratio: float = 0.35
    ruler_min_width_px: int = 18
    ruler_max_width_ratio: float = 0.42
    ruler_pad_px: int = 4
    ruler_tick_dark_threshold: int = 105
    ruler_pale_v_min: int = 90
    ruler_pale_s_max: int = 150
    ruler_min_tick_rows: int = 6

    # Fallback manual opcional para regua, se necessario.
    manual_ruler_roi: Optional[Tuple[int, int, int, int]] = None
    manual_ruler_top_y_px: float = 0.0
    manual_ruler_zero_y_px: float = 0.0

    # Bolinha preta/escura: busca no frame inteiro, excluindo regua e LED vermelho.
    ball_dark_threshold: int = 120
    # V20: piso do threshold efetivo. Evita voltar para 120 quando o percentil escuro fica baixo demais
    # e a mascara engole fundo/sombra/base. Threshold final = min(ball_dark_threshold, max(qthr, ball_min_effective_threshold)).
    ball_min_effective_threshold: int = 75
    ball_dark_percentile: float = 8.0
    ball_mask_open_px: int = 3
    ball_mask_close_px: int = 13
    ball_min_area_px: int = 6000
    ball_max_area_ratio: float = 0.42
    ball_min_width_px: int = 80
    ball_min_height_px: int = 70
    ball_max_width_ratio: float = 0.70
    ball_max_height_ratio: float = 0.70
    ball_min_fill_ratio: float = 0.08
    ball_min_circularity: float = 0.03
    ball_max_aspect_ratio: float = 2.20       # rejeita fenda/base horizontal; bolinha real tende a ser mais alta
    ball_reject_border_px: int = 3
    reject_giant_border_blobs: bool = True
    giant_blob_width_ratio: float = 0.82
    giant_blob_height_ratio: float = 0.72
    giant_blob_area_ratio: float = 0.24
    giant_blob_aspect_ratio: float = 2.60
    ruler_exclusion_margin_px: int = 8
    led_exclusion_dilate_px: int = 15

    # V19 - crop vertical opcional da busca da bolinha.
    # Mantem apenas a faixa [ball_crop_ymin_ratio, ball_crop_ymax_ratio] do frame
    # para a segmentacao da bolinha. A regua e o LED continuam no frame inteiro.
    # Exemplo: --ball-crop-ymax-ratio 0.70 ignora os 30% inferiores, util para
    # remover buraco/sombra/base sem mexer no video original.
    ball_crop_ymin_ratio: float = 0.0
    ball_crop_ymax_ratio: float = 1.0

    # Medicao do topo do blob.
    top_percentile: float = 6.0               # percentil baixo dos pixels superiores por coluna
    top_min_column_height_px: int = 8
    top_refine_subpixel: bool = True
    top_refine_window_px: int = 5
    top_smooth_window: int = 7

    # Associacao temporal simples.
    max_jump_px_soft: float = 65.0
    jump_penalty_weight: float = 0.012
    missing_penalty: float = 1.20
    max_candidates_per_frame: int = 8
    valid_score_threshold: float = 0.12

    # Sincronismo LED vermelho.
    sync_enabled: bool = True
    sync_crop_to_rising_edge: bool = True
    sync_red_min_area_px: int = 12
    sync_score_threshold: float = 0.45
    sync_temporal_smooth_frames: int = 1
    sync_known_pulse_width_ms: float = 200.0
    sync_min_pulse_width_frames_for_fps: int = 6

    # Preview/plot.
    preview_frames: int = 18
    preview_thumb_width_px: int = 360


# ============================================================
# Utilitarios
# ============================================================

def parse_tuple4(s: str) -> Tuple[int, int, int, int]:
    vals = [int(float(v.strip())) for v in s.split(",")]
    if len(vals) != 4:
        raise ValueError("Esperado formato x,y,w,h")
    return vals[0], vals[1], vals[2], vals[3]


def odd_at_least(value: int, minimum: int = 3) -> int:
    v = max(int(value), int(minimum))
    if v % 2 == 0:
        v += 1
    return v


def clamp_roi(roi: Tuple[int, int, int, int], W: int, H: int, border: int = 0) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(border, min(int(x), W - 1 - border))
    y = max(border, min(int(y), H - 1 - border))
    x2 = max(x + 1, min(int(x + w), W - border))
    y2 = max(y + 1, min(int(y + h), H - border))
    return int(x), int(y), int(x2 - x), int(y2 - y)


def _parse_ratio(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s == "0/0":
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            a = float(a)
            b = float(b)
            return a / b if b else 0.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _ffprobe_json(video_path: str) -> Optional[dict]:
    if shutil.which("ffprobe") is None:
        return None
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=12)
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        return json.loads(proc.stdout)
    except Exception:
        return None


def _find_capture_fps_in_ffprobe(ffdata: Optional[dict]) -> Tuple[float, str]:
    if not ffdata:
        return 0.0, "none"
    keys = [
        "com.android.capture.fps", "com.android.capture.framerate",
        "com.samsung.android.capture.fps", "captureframerate", "capture_framerate",
        "originalframerate", "original_frame_rate", "slow_motion_capture_fps",
    ]
    containers = []
    if isinstance(ffdata.get("format"), dict):
        containers.append(ffdata["format"].get("tags", {}) or {})
    for stream in ffdata.get("streams", []) or []:
        if isinstance(stream, dict):
            containers.append(stream.get("tags", {}) or {})
            containers.append(stream)
    for d in containers:
        lowered = {str(k).lower(): v for k, v in d.items()}
        for key in keys:
            if key in lowered:
                fps = _parse_ratio(lowered[key])
                if fps > 1:
                    return fps, f"ffprobe_tag:{key}"
    return 0.0, "none"


def read_video_metadata(video_path: str) -> Dict[str, object]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nao consegui abrir o video: {video_path}")
    fps_cv = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    fps_playback = fps_cv
    fps_capture = 0.0
    source = "opencv_playback"
    confidence = "LOW"

    ffdata = _ffprobe_json(video_path)
    if ffdata:
        for stream in ffdata.get("streams", []) or []:
            if stream.get("codec_type") == "video":
                avg = _parse_ratio(stream.get("avg_frame_rate"))
                rate = _parse_ratio(stream.get("r_frame_rate"))
                if avg > 1:
                    fps_playback = avg
                    source = "ffprobe_stream_avg_frame_rate"
                elif rate > 1:
                    fps_playback = rate
                    source = "ffprobe_stream_r_frame_rate"
                if n <= 0:
                    try:
                        n = int(float(stream.get("nb_frames", 0)))
                    except Exception:
                        pass
                break
        fps_tag, tag_source = _find_capture_fps_in_ffprobe(ffdata)
        if fps_tag > 1:
            fps_capture = fps_tag
            source = tag_source
            confidence = "HIGH"

    if fps_capture <= 0:
        fps_capture = fps_playback
    return {
        "fps": float(fps_playback),
        "fps_playback": float(fps_playback),
        "fps_capture": float(fps_capture),
        "frame_count": int(n),
        "width": int(W),
        "height": int(H),
        "metadata_time_source": source,
        "metadata_time_confidence": confidence,
        "time_scale_detected": float(fps_playback / fps_capture) if fps_capture else np.nan,
    }


def sample_frame(video_path: str, frame_idx: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Nao consegui ler o frame {frame_idx}")
    return frame


def iter_video_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


# ============================================================
# Sincronismo LED vermelho
# ============================================================

def red_led_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(frame)
    mask1 = cv2.inRange(hsv, (0, 80, 70), (12, 255, 255))
    mask2 = cv2.inRange(hsv, (168, 80, 70), (180, 255, 255))
    red_dom = (
        (r.astype(np.int16) > 95)
        & (r.astype(np.int16) > g.astype(np.int16) + 28)
        & (r.astype(np.int16) > b.astype(np.int16) + 28)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_and(cv2.bitwise_or(mask1, mask2), red_dom)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def red_led_score(frame: np.ndarray, p: Params) -> dict:
    mask = red_led_mask(frame)
    b, g, r = cv2.split(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = {
        "sync_red_area_px": 0.0,
        "sync_red_centroid_x_px": np.nan,
        "sync_red_centroid_y_px": np.nan,
        "sync_raw_score": 0.0,
    }
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < p.sync_red_min_area_px:
            continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-9:
            continue
        cmask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(cmask, [c], -1, 255, -1)
        mean_r = float(cv2.mean(r, mask=cmask)[0])
        mean_g = float(cv2.mean(g, mask=cmask)[0])
        mean_b = float(cv2.mean(b, mask=cmask)[0])
        red_gain = max(0.0, mean_r - 0.5 * (mean_g + mean_b))
        score = area * red_gain / 255.0
        if score > best["sync_raw_score"]:
            best = {
                "sync_red_area_px": area,
                "sync_red_centroid_x_px": float(M["m10"] / M["m00"]),
                "sync_red_centroid_y_px": float(M["m01"] / M["m00"]),
                "sync_raw_score": float(score),
            }
    return best


def finalize_sync_dataframe(sync_df: pd.DataFrame, p: Params) -> Tuple[pd.DataFrame, int, int]:
    if sync_df.empty:
        return sync_df, -1, -1
    raw = sync_df["sync_raw_score"].to_numpy(dtype=float)
    lo = float(np.nanmedian(raw)) if np.isfinite(raw).any() else 0.0
    hi = float(np.nanpercentile(raw, 99.0)) if np.isfinite(raw).any() else 0.0
    if hi <= lo:
        hi = float(np.nanmax(raw)) if np.isfinite(raw).any() else 0.0
    score = np.zeros_like(raw, dtype=float) if hi <= lo else np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    pulse = score >= float(p.sync_score_threshold)

    # fecha buracos/ilhas curtissimas
    smooth_iters = max(0, int(p.sync_temporal_smooth_frames))
    pulse_i = pulse.astype(np.uint8)
    for _ in range(smooth_iters):
        for i in range(1, len(pulse_i) - 1):
            if pulse_i[i - 1] and pulse_i[i + 1]:
                pulse_i[i] = 1
        for i in range(1, len(pulse_i) - 1):
            if not pulse_i[i - 1] and not pulse_i[i + 1]:
                pulse_i[i] = 0
    pulse = pulse_i.astype(bool)

    rising = np.zeros(len(pulse), dtype=bool)
    falling = np.zeros(len(pulse), dtype=bool)
    prev = False
    for i, val in enumerate(pulse):
        if val and not prev:
            rising[i] = True
        if (not val) and prev:
            falling[i] = True
        prev = bool(val)

    rise_idxs = np.flatnonzero(rising)
    fall_idxs = np.flatnonzero(falling)
    rise_frame = int(rise_idxs[0]) if len(rise_idxs) else -1
    fall_frame = -1
    if rise_frame >= 0:
        after = fall_idxs[fall_idxs > rise_frame]
        if len(after):
            fall_frame = int(after[0])

    events = []
    for i in range(len(pulse)):
        if rising[i]:
            events.append("RISING_EDGE")
        elif falling[i]:
            events.append("FALLING_EDGE")
        elif pulse[i]:
            events.append("HIGH")
        else:
            events.append("LOW")

    out = sync_df.copy()
    out["sync_score"] = score
    out["sync_pulse"] = pulse.astype(int)
    out["sync_event"] = events
    out["sync_rising_edge"] = rising.astype(int)
    out["sync_falling_edge"] = falling.astype(int)
    out["sync_rising_frame"] = rise_frame
    out["sync_falling_frame"] = fall_frame
    out["sync_pulse_width_frames"] = (fall_frame - rise_frame) if (rise_frame >= 0 and fall_frame >= 0) else np.nan
    return out, rise_frame, fall_frame


def collect_sync(video_path: str, p: Params) -> Tuple[pd.DataFrame, int, int]:
    if not p.sync_enabled:
        return pd.DataFrame(), -1, -1
    rows = []
    for idx, frame in iter_video_frames(video_path):
        row = {"frame": int(idx)}
        row.update(red_led_score(frame, p))
        rows.append(row)
    return finalize_sync_dataframe(pd.DataFrame(rows), p)


# ============================================================
# Regua fixa reta de 26 mm
# ============================================================

def find_row_peaks(row_score: np.ndarray, min_distance: int = 4) -> List[int]:
    if row_score.size == 0:
        return []
    smooth = np.convolve(row_score.astype(float), np.ones(3) / 3.0, mode="same")
    thr = max(float(np.percentile(smooth, 80.0)), float(np.mean(smooth) + 0.25 * np.std(smooth)), 1.0)
    cand = np.flatnonzero(smooth >= thr)
    peaks: List[int] = []
    for i in cand:
        i = int(i)
        lo = max(0, i - 2)
        hi = min(len(smooth), i + 3)
        if smooth[i] < np.max(smooth[lo:hi]) - 1e-9:
            continue
        if peaks and i - peaks[-1] < min_distance:
            if smooth[i] > smooth[peaks[-1]]:
                peaks[-1] = i
        else:
            peaks.append(i)
    return peaks


def detect_fixed_ruler(frame: np.ndarray, p: Params) -> dict:
    H, W = frame.shape[:2]

    if p.manual_ruler_roi is not None:
        x, y, w, h = clamp_roi(p.manual_ruler_roi, W, H, 0)
        top_y = float(p.manual_ruler_top_y_px) if p.manual_ruler_top_y_px > 0 else float(y)
        zero_y = float(p.manual_ruler_zero_y_px) if p.manual_ruler_zero_y_px > 0 else float(y + h)
        px_per_mm = (zero_y - top_y) / float(p.ruler_height_mm)
        return {
            "ruler_ok": bool(px_per_mm > 0),
            "ruler_reason": "manual_roi",
            "ruler_roi_x_px": x, "ruler_roi_y_px": y, "ruler_roi_w_px": w, "ruler_roi_h_px": h,
            "ruler_top_y_px": top_y, "ruler_zero_y_px": zero_y,
            "ruler_top_x_px": float(x + w / 2), "ruler_zero_x_px": float(x + w / 2),
            "ruler_height_mm": float(p.ruler_height_mm),
            "px_per_mm": float(px_per_mm), "mm_per_px": float(1.0 / px_per_mm) if px_per_mm > 0 else np.nan,
            "ruler_tick_count": np.nan,
        }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    x_search0 = int(W * float(p.ruler_search_x0_ratio))
    x_search0 = max(0, min(W - 2, x_search0))

    pale = ((val > int(p.ruler_pale_v_min)) & (sat < int(p.ruler_pale_s_max))).astype(np.uint8) * 255
    pale[:, :x_search0] = 0
    pale = cv2.morphologyEx(pale, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 25)), iterations=2)

    contours, _ = cv2.findContours(pale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < p.ruler_min_height_ratio * H:
            continue
        if w < p.ruler_min_width_px or w > p.ruler_max_width_ratio * W:
            continue
        if x < x_search0 - 5:
            continue
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue
        dark = (roi_gray < int(p.ruler_tick_dark_threshold)).astype(np.uint8) * 255
        # prioriza marcas horizontais na metade esquerda da regua
        strip_w = max(8, int(0.55 * w))
        strip = dark[:, :strip_w]
        row_score = strip.sum(axis=1).astype(float) / 255.0
        peaks = find_row_peaks(row_score, min_distance=4)
        tick_count = len(peaks)
        score = 1000.0 * tick_count + 0.05 * h + 0.02 * w + 0.01 * x
        candidates.append((score, tick_count, x, y, w, h, peaks))

    if not candidates:
        # fallback simples: faixa direita completa; ainda gera preview para diagnostico
        x = int(W * 0.66)
        y = int(H * 0.05)
        w = int(W * 0.25)
        h = int(H * 0.80)
        x, y, w, h = clamp_roi((x, y, w, h), W, H, 0)
        top_y = float(y)
        zero_y = float(y + h)
        px_per_mm = (zero_y - top_y) / float(p.ruler_height_mm)
        return {
            "ruler_ok": False,
            "ruler_reason": "fallback_no_component",
            "ruler_roi_x_px": x, "ruler_roi_y_px": y, "ruler_roi_w_px": w, "ruler_roi_h_px": h,
            "ruler_top_y_px": top_y, "ruler_zero_y_px": zero_y,
            "ruler_top_x_px": float(x + w / 2), "ruler_zero_x_px": float(x + w / 2),
            "ruler_height_mm": float(p.ruler_height_mm),
            "px_per_mm": float(px_per_mm), "mm_per_px": float(1.0 / px_per_mm) if px_per_mm > 0 else np.nan,
            "ruler_tick_count": 0,
        }

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, tick_count, x, y, w, h, peaks = candidates[0]
    pad = int(p.ruler_pad_px)
    x, y, w, h = clamp_roi((x - pad, y - pad, w + 2 * pad, h + 2 * pad), W, H, 0)

    # Como a regua fisica foi retificada e tem 26 mm do zero ao topo,
    # usa o corpo visivel da regua como span metrologico.
    top_y = float(y)
    zero_y = float(y + h)
    px_per_mm = (zero_y - top_y) / float(p.ruler_height_mm)
    ok = bool(px_per_mm > 0 and tick_count >= int(p.ruler_min_tick_rows))
    return {
        "ruler_ok": ok,
        "ruler_reason": "OK" if ok else "few_ticks_body_detected",
        "ruler_roi_x_px": int(x), "ruler_roi_y_px": int(y), "ruler_roi_w_px": int(w), "ruler_roi_h_px": int(h),
        "ruler_top_y_px": top_y, "ruler_zero_y_px": zero_y,
        "ruler_top_x_px": float(x + w / 2), "ruler_zero_x_px": float(x + w / 2),
        "ruler_height_mm": float(p.ruler_height_mm),
        "px_per_mm": float(px_per_mm), "mm_per_px": float(1.0 / px_per_mm) if px_per_mm > 0 else np.nan,
        "ruler_tick_count": int(tick_count),
    }


# ============================================================
# Bolinha escura: maior blob plausivel + topo direto
# ============================================================

def exclusion_mask_for_ruler_and_led(frame: np.ndarray, ruler: dict, p: Params) -> np.ndarray:
    H, W = frame.shape[:2]
    allowed = np.ones((H, W), dtype=np.uint8) * 255

    # Crop vertical opcional apenas para a bolinha.
    # y cresce para baixo: ymin=0.0, ymax=0.70 mantem os 70% superiores.
    y_min_crop, y_max_crop = get_ball_crop_limits(H, p)
    if y_min_crop > 0:
        allowed[:y_min_crop, :] = 0
    if y_max_crop < H:
        allowed[y_max_crop:, :] = 0

    # Exclui a regua, mas so se a deteccao tiver ROI razoavel.
    x = int(ruler.get("ruler_roi_x_px", W + 1))
    y = int(ruler.get("ruler_roi_y_px", 0))
    w = int(ruler.get("ruler_roi_w_px", 0))
    h = int(ruler.get("ruler_roi_h_px", 0))
    if w > 0 and h > 0:
        m = int(p.ruler_exclusion_margin_px)
        x1 = max(0, x - m)
        y1 = max(0, y - m)
        x2 = min(W, x + w + m)
        y2 = min(H, y + h + m)
        allowed[y1:y2, x1:x2] = 0

    # Exclui o LED vermelho para ele nunca virar candidato escuro/local.
    red = red_led_mask(frame)
    d = int(p.led_exclusion_dilate_px)
    if d > 0:
        red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_at_least(d), odd_at_least(d))), iterations=1)
    allowed[red > 0] = 0
    return allowed




def get_ball_crop_limits(H: int, p: Params) -> Tuple[int, int]:
    ymin_ratio = float(np.clip(getattr(p, "ball_crop_ymin_ratio", 0.0), 0.0, 1.0))
    ymax_ratio = float(np.clip(getattr(p, "ball_crop_ymax_ratio", 1.0), 0.0, 1.0))
    if ymax_ratio <= ymin_ratio:
        ymin_ratio, ymax_ratio = 0.0, 1.0
    y_min_crop = int(round(ymin_ratio * H))
    y_max_crop = int(round(ymax_ratio * H))
    y_min_crop = max(0, min(H, y_min_crop))
    y_max_crop = max(y_min_crop + 1, min(H, y_max_crop))
    return y_min_crop, y_max_crop


def compute_dark_threshold(gray_blur: np.ndarray, allowed: np.ndarray, p: Params) -> Tuple[float, float]:
    valid_pixels = gray_blur[allowed > 0]
    abs_thr = float(p.ball_dark_threshold)
    min_eff = float(getattr(p, "ball_min_effective_threshold", 75))
    if valid_pixels.size:
        qthr = float(np.percentile(valid_pixels, float(p.ball_dark_percentile)))
        thr = min(abs_thr, max(qthr, min_eff))
    else:
        qthr = float("nan")
        thr = abs_thr
    return float(thr), float(qthr)

def make_dark_ball_mask(frame: np.ndarray, ruler: dict, p: Params) -> np.ndarray:
    """Mascara simples da bolinha preta.

    A V16 estava permissiva demais: usava max(threshold_absoluto, percentil) e
    ainda fazia OR com adaptiveThreshold. Em cena clara isso transforma sombra,
    papel e fundo em um blob gigante; depois os filtros rejeitam tudo e o preview
    fica INVALID.

    A V17 volta ao simples: threshold absoluto de escuro, com um limite opcional
    por percentil apenas para apertar, nunca para afrouxar.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    allowed = exclusion_mask_for_ruler_and_led(frame, ruler, p)

    thr, _qthr = compute_dark_threshold(gray_blur, allowed, p)

    mask = (gray_blur <= thr).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, allowed)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_at_least(p.ball_mask_open_px), odd_at_least(p.ball_mask_open_px)))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_at_least(p.ball_mask_close_px), odd_at_least(p.ball_mask_close_px)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    return mask

def circularity(contour) -> float:
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri <= 1e-9:
        return 0.0
    return float(4.0 * math.pi * area / (peri * peri))


def measure_top_from_component(mask: np.ndarray, contour, frame: np.ndarray, p: Params) -> dict:
    x, y, w, h = cv2.boundingRect(contour)
    comp = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(comp, [contour], -1, 255, -1)

    ys_top = []
    xs_used = []
    x1 = max(0, x)
    x2 = min(mask.shape[1], x + w)
    for xx in range(x1, x2):
        col = np.flatnonzero(comp[:, xx] > 0)
        if col.size < int(p.top_min_column_height_px):
            continue
        ys_top.append(float(col.min()))
        xs_used.append(float(xx))

    if len(ys_top) < 4:
        return {"top_valid": False, "top_reason": "few_columns"}

    ys_top_arr = np.asarray(ys_top, dtype=float)
    xs_arr = np.asarray(xs_used, dtype=float)
    y_int = float(np.percentile(ys_top_arr, float(p.top_percentile)))

    # Refinamento subpixel: perto da borda superior, procura gradiente claro->escuro.
    y_sub = y_int
    if p.top_refine_subpixel:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        near = ys_top_arr <= (y_int + max(3.0, 0.06 * h))
        x_cols = xs_arr[near].astype(int)
        refined = []
        half = int(p.top_refine_window_px)
        for xx in x_cols[::max(1, len(x_cols) // 80)]:
            yy = int(round(y_int))
            ylo = max(1, yy - half)
            yhi = min(gray.shape[0] - 2, yy + half)
            if yhi <= ylo + 2:
                continue
            prof = gray[ylo:yhi + 1, xx]
            # objeto escuro: ao descer, intensidade cai; usa -gradiente.
            grad = -(prof[2:] - prof[:-2])
            if grad.size == 0:
                continue
            k = int(np.argmax(grad)) + ylo + 1
            if k <= 0 or k >= gray.shape[0] - 1:
                continue
            gm = float(max(0.0, -(gray[k, xx] - gray[k - 2, xx]) if k >= 2 else 0.0))
            g0 = float(max(0.0, -(gray[k + 1, xx] - gray[k - 1, xx])))
            gp = float(max(0.0, -(gray[k + 2, xx] - gray[k, xx]) if k + 2 < gray.shape[0] else 0.0))
            denom = gm - 2.0 * g0 + gp
            off = 0.0 if abs(denom) < 1e-9 else float(np.clip(0.5 * (gm - gp) / denom, -0.75, 0.75))
            refined.append(float(k + off))
        if len(refined) >= 5:
            y_sub = float(np.median(refined))

    return {
        "top_valid": True,
        "top_reason": "OK",
        "y_top_blob_px": float(y_int),
        "y_top_subpx": float(y_sub),
        "x_top_blob_px": float(np.median(xs_arr[ys_top_arr <= y_int + max(3.0, 0.06 * h)])),
        "top_columns_used": int(len(ys_top_arr)),
    }



def rejection_message(reason: str) -> str:
    mensagens = {
        "too_small_height": "Objeto escuro rejeitado porque é baixo demais para ser a bolinha. Provavelmente é a fenda, a sombra da base ou um pedaço de papel.",
        "too_small_width": "Objeto escuro rejeitado porque é estreito demais para ser a bolinha.",
        "too_small_area": "Objeto escuro rejeitado porque a área é pequena demais para ser a bolinha.",
        "too_large_area": "Objeto escuro rejeitado porque a área é grande demais. Provavelmente a bolinha grudou visualmente com fundo, sombra ou base.",
        "too_wide": "Objeto escuro rejeitado porque ficou largo demais para ser a bolinha no enquadramento atual.",
        "too_tall": "Objeto escuro rejeitado porque ficou alto demais para ser a bolinha no enquadramento atual.",
        "too_flat_aspect": "Objeto escuro rejeitado porque está achatado horizontalmente. Isso parece mais uma fenda/base do que a bolinha.",
        "giant_border_blob": "Objeto escuro rejeitado porque virou um bloco gigante encostado na borda ou no limite do crop. Provavelmente é fundo/base grudado na máscara.",
        "bad_fill": "Objeto escuro rejeitado porque o preenchimento do retângulo é baixo demais; o formato está esparso ou fragmentado.",
        "bad_circularity": "Objeto escuro rejeitado porque o contorno está irregular demais para a bolinha.",
        "few_top_columns": "Objeto escuro rejeitado porque não havia colunas suficientes para medir o topo com segurança.",
    }
    return mensagens.get(reason, "Objeto escuro rejeitado por critério interno de validação.")


def make_rejection_row(frame_idx: int, component_id: int, reason: str, contour, area: float, W: int, H: int, p: Params, mask_threshold: float, q_dark: float) -> dict:
    x, y, w, h = cv2.boundingRect(contour)
    fill = area / max(1.0, float(w * h))
    circ = circularity(contour) if area > 0 else 0.0
    aspect = float(w) / max(1.0, float(h))
    y_min_crop, y_max_crop = get_ball_crop_limits(H, p)
    touches_crop = bool(y <= y_min_crop + int(p.ball_reject_border_px) or (y + h) >= y_max_crop - int(p.ball_reject_border_px))
    touches_border = bool(x <= int(p.ball_reject_border_px) or (x + w) >= W - int(p.ball_reject_border_px))
    return {
        "frame": int(frame_idx),
        "componente": int(component_id),
        "problema": rejection_message(reason),
        "codigo_problema": reason,
        "area_px": float(area),
        "largura_px": float(w),
        "altura_px": float(h),
        "x_px": float(x),
        "y_px": float(y),
        "proporcao_largura_altura": float(aspect),
        "preenchimento": float(fill),
        "circularidade": float(circ),
        "tocou_crop_vertical": touches_crop,
        "tocou_borda_lateral": touches_border,
        "threshold_usado": float(mask_threshold),
        "percentil_escuro_q": float(q_dark) if np.isfinite(q_dark) else np.nan,
        "min_altura_config_px": float(p.ball_min_height_px),
        "min_largura_config_px": float(p.ball_min_width_px),
        "min_area_config_px": float(p.ball_min_area_px),
    }


def detect_ball_candidates(frame: np.ndarray, ruler: dict, p: Params, frame_idx: int = 0) -> Tuple[List[dict], np.ndarray, List[dict], dict]:
    H, W = frame.shape[:2]
    mask = make_dark_ball_mask(frame, ruler, p)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    allowed = exclusion_mask_for_ruler_and_led(frame, ruler, p)
    mask_threshold, q_dark = compute_dark_threshold(gray_blur, allowed, p)

    max_area = float(p.ball_max_area_ratio) * float(W * H)
    candidates: List[dict] = []
    rejection_rows: List[dict] = []
    y_min_crop, y_max_crop = get_ball_crop_limits(H, p)
    crop_h = max(1, y_max_crop - y_min_crop)

    for component_id, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        aspect = float(w) / max(1.0, float(h))

        # Ordem proposital: altura/largura primeiro para o log explicar claramente
        # quando a fenda/base está competindo com a bolinha.
        if h < p.ball_min_height_px:
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "too_small_height", contour, area, W, H, p, mask_threshold, q_dark))
            continue
        if w < p.ball_min_width_px:
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "too_small_width", contour, area, W, H, p, mask_threshold, q_dark))
            continue
        if aspect > float(getattr(p, "ball_max_aspect_ratio", 2.20)):
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "too_flat_aspect", contour, area, W, H, p, mask_threshold, q_dark))
            continue
        if area < float(p.ball_min_area_px):
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "too_small_area", contour, area, W, H, p, mask_threshold, q_dark))
            continue
        if area > max_area:
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "too_large_area", contour, area, W, H, p, mask_threshold, q_dark))
            continue
        if w > p.ball_max_width_ratio * W:
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "too_wide", contour, area, W, H, p, mask_threshold, q_dark))
            continue
        if h > p.ball_max_height_ratio * H:
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "too_tall", contour, area, W, H, p, mask_threshold, q_dark))
            continue

        b = int(p.ball_reject_border_px)
        touches_border = (
            x <= b or (x + w) >= (W - b)
            or y <= (y_min_crop + b)
            or (y + h) >= (y_max_crop - b)
        )
        is_giant = (
            w >= float(getattr(p, "giant_blob_width_ratio", 0.82)) * W
            or h >= float(getattr(p, "giant_blob_height_ratio", 0.72)) * crop_h
            or area >= float(getattr(p, "giant_blob_area_ratio", 0.24)) * W * H
            or aspect >= float(getattr(p, "giant_blob_aspect_ratio", 2.60))
        )
        if bool(getattr(p, "reject_giant_border_blobs", True)) and touches_border and is_giant:
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "giant_border_blob", contour, area, W, H, p, mask_threshold, q_dark))
            continue

        fill = area / max(1.0, float(w * h))
        if fill < float(p.ball_min_fill_ratio):
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "bad_fill", contour, area, W, H, p, mask_threshold, q_dark))
            continue
        circ = circularity(contour)
        if circ < float(p.ball_min_circularity):
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "bad_circularity", contour, area, W, H, p, mask_threshold, q_dark))
            continue

        cmask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(cmask, [contour], -1, 255, -1)
        mean_gray = float(cv2.mean(gray, mask=cmask)[0])
        darkness = float(np.clip((160.0 - mean_gray) / 160.0, 0.0, 1.0))
        area_score = float(np.clip(area / max(1.0, 0.10 * W * H), 0.0, 1.0))
        fill_score = float(np.clip((fill - p.ball_min_fill_ratio) / max(1e-9, 0.90 - p.ball_min_fill_ratio), 0.0, 1.0))
        circ_score = float(np.clip(circ, 0.0, 1.0))

        M = cv2.moments(contour)
        if abs(M["m00"]) > 1e-9:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = float(x + w / 2)
            cy = float(y + h / 2)

        top_res = measure_top_from_component(mask, contour, frame, p)
        if not top_res.get("top_valid", False):
            rejection_rows.append(make_rejection_row(frame_idx, component_id, "few_top_columns", contour, area, W, H, p, mask_threshold, q_dark))
            continue

        score = 0.34 * area_score + 0.30 * darkness + 0.18 * fill_score + 0.12 * circ_score + 0.06
        cand = {
            "x_center_px": cx,
            "y_center_px": cy,
            "bbox_x_px": float(x),
            "bbox_y_px": float(y),
            "bbox_w_px": float(w),
            "bbox_h_px": float(h),
            "bbox_aspect": float(aspect),
            "area_px": area,
            "fill_ratio": float(fill),
            "circularity": float(circ),
            "mean_gray": mean_gray,
            "confidence": float(np.clip(score, 0.0, 1.0)),
            "method": "dark_blob",
            "mask_threshold": float(mask_threshold),
            "q_dark": float(q_dark) if np.isfinite(q_dark) else np.nan,
        }
        cand.update(top_res)
        candidates.append(cand)

    candidates.sort(key=lambda d: d["confidence"], reverse=True)
    frame_debug = {
        "frame": int(frame_idx),
        "componentes_encontrados_na_mascara": int(len(contours)),
        "candidatos_aceitos": int(min(len(candidates), int(p.max_candidates_per_frame))),
        "componentes_rejeitados": int(len(rejection_rows)),
        "threshold_usado": float(mask_threshold),
        "percentil_escuro_q": float(q_dark) if np.isfinite(q_dark) else np.nan,
    }
    return candidates[: int(p.max_candidates_per_frame)], mask, rejection_rows, frame_debug

def choose_track_dp(all_candidates: List[List[dict]], p: Params) -> List[Optional[dict]]:
    n = len(all_candidates)
    if n == 0:
        return []
    costs: List[List[float]] = []
    prevs: List[List[int]] = []
    for i, cands0 in enumerate(all_candidates):
        cands = cands0 if cands0 else [None]
        frame_costs = []
        frame_prevs = []
        for cand in cands:
            det_cost = float(p.missing_penalty) if cand is None else (1.0 - float(cand["confidence"]))
            if i == 0:
                frame_costs.append(det_cost)
                frame_prevs.append(-1)
                continue
            best = float("inf")
            best_k = -1
            pcands = all_candidates[i - 1] if all_candidates[i - 1] else [None]
            for k, pcand in enumerate(pcands):
                prev_cost = costs[i - 1][k]
                if cand is None or pcand is None:
                    jump_cost = 0.20
                else:
                    dx = float(cand["x_center_px"]) - float(pcand["x_center_px"])
                    dy = float(cand["y_center_px"]) - float(pcand["y_center_px"])
                    dist = math.hypot(dx, dy)
                    excess = max(0.0, dist - float(p.max_jump_px_soft))
                    jump_cost = float(p.jump_penalty_weight) * (dist + 3.0 * excess)
                total = prev_cost + det_cost + jump_cost
                if total < best:
                    best = total
                    best_k = k
            frame_costs.append(best)
            frame_prevs.append(best_k)
        costs.append(frame_costs)
        prevs.append(frame_prevs)

    idx = int(np.argmin(costs[-1]))
    track: List[Optional[dict]] = [None] * n
    for i in range(n - 1, -1, -1):
        cands = all_candidates[i] if all_candidates[i] else [None]
        track[i] = cands[idx]
        idx = prevs[i][idx]
        if idx < 0 and i > 0:
            idx = 0
    return track


# ============================================================
# Processamento principal e saidas
# ============================================================

def add_time_columns(df: pd.DataFrame, meta: Dict[str, object], sync_df: pd.DataFrame, rise_frame: int, fall_frame: int, p: Params) -> pd.DataFrame:
    df = df.copy()
    if not sync_df.empty:
        df = df.merge(sync_df, on="frame", how="left")
    else:
        for c in ["sync_score", "sync_pulse", "sync_event", "sync_rising_edge", "sync_falling_edge", "sync_rising_frame", "sync_falling_frame"]:
            df[c] = np.nan

    fps_playback = float(meta.get("fps_playback", meta.get("fps", 0.0)) or 0.0)
    fps_capture = float(meta.get("fps_capture", fps_playback) or 0.0)
    source = str(meta.get("metadata_time_source", "unknown"))
    conf = str(meta.get("metadata_time_confidence", "LOW"))

    pulse_frames = (fall_frame - rise_frame) if (rise_frame >= 0 and fall_frame > rise_frame) else 0
    if conf == "LOW" and pulse_frames >= int(p.sync_min_pulse_width_frames_for_fps) and p.sync_known_pulse_width_ms > 0:
        fps_capture = 1000.0 * float(pulse_frames) / float(p.sync_known_pulse_width_ms)
        conf = "PULSE_ESTIMATED"
        source = "sync_led_known_width"
    elif conf == "LOW" and rise_frame >= 0 and fall_frame > rise_frame:
        conf = "LOW_SYNC_WIDTH"
        source = "sync_led_width_rejected"

    df["time_video_ms"] = (df["frame"] / fps_playback * 1000.0) if fps_playback else np.nan
    df["time_real_ms"] = (df["frame"] / fps_capture * 1000.0) if fps_capture else np.nan
    df["frame_sync"] = df["frame"] - int(rise_frame) if rise_frame >= 0 else np.nan
    df["time_sync_ms"] = (df["frame_sync"] / fps_capture * 1000.0) if (rise_frame >= 0 and fps_capture) else np.nan
    df["time_from_sync_fall_ms"] = ((df["frame"] - int(fall_frame)) / fps_capture * 1000.0) if (fall_frame >= 0 and fps_capture) else np.nan
    df["fps_playback"] = fps_playback
    df["fps_capture"] = fps_capture
    df["metadata_time_source"] = source
    df["metadata_time_confidence"] = conf
    df["time_scale_detected"] = (fps_playback / fps_capture) if fps_capture else np.nan
    df["sync_pulse_width_frames"] = pulse_frames if pulse_frames > 0 else np.nan
    df["sync_pulse_width_ms"] = (pulse_frames / fps_capture * 1000.0) if (pulse_frames > 0 and fps_capture) else np.nan
    return df


def process_video(video_path: str, output_dir: str, p: Params):
    os.makedirs(output_dir, exist_ok=True)
    meta = read_video_metadata(video_path)
    W, H = int(meta["width"]), int(meta["height"])
    n = int(meta["frame_count"])
    fps = float(meta["fps"])

    # Regua fixa detectada uma vez no primeiro frame.
    first = sample_frame(video_path, 0)
    ruler = detect_fixed_ruler(first, p)

    # LED de sync antes do tracking, para cortar depois.
    sync_df, rise_frame, fall_frame = collect_sync(video_path, p)

    all_candidates: List[List[dict]] = []
    candidate_rows: List[dict] = []
    rejection_rows: List[dict] = []
    frame_debug_rows: List[dict] = []
    frame_ids: List[int] = []
    last_mask_by_frame: Dict[int, np.ndarray] = {}
    preview_idx_set = set(np.linspace(0, max(0, n - 1), min(p.preview_frames, max(1, n))).astype(int))

    for idx, frame in iter_video_frames(video_path):
        cands, mask, rejected, frame_debug = detect_ball_candidates(frame, ruler, p, frame_idx=int(idx))
        all_candidates.append(cands)
        rejection_rows.extend(rejected)
        frame_debug_rows.append(frame_debug)
        frame_ids.append(int(idx))
        if idx in preview_idx_set:
            last_mask_by_frame[int(idx)] = mask
        for rank, c in enumerate(cands):
            row = {"frame": int(idx), "rank": int(rank)}
            row.update(c)
            candidate_rows.append(row)

    track = choose_track_dp(all_candidates, p)

    rows = []
    decision_rows: List[dict] = []
    error_lines: List[str] = []
    px_per_mm = float(ruler.get("px_per_mm", np.nan))
    zero_y = float(ruler.get("ruler_zero_y_px", np.nan))
    for i, cand in enumerate(track):
        frame = int(frame_ids[i]) if i < len(frame_ids) else i
        base = {
            "frame": frame,
            "valid": False,
            "method": "missing",
            "confidence": 0.0,
            "x_center_px": np.nan,
            "y_center_px": np.nan,
            "bbox_x_px": np.nan,
            "bbox_y_px": np.nan,
            "bbox_w_px": np.nan,
            "bbox_h_px": np.nan,
            "area_px": np.nan,
            "fill_ratio": np.nan,
            "circularity": np.nan,
            "mean_gray": np.nan,
            "y_top_blob_px": np.nan,
            "y_top_subpx": np.nan,
            "x_top_blob_px": np.nan,
            "top_columns_used": 0,
            "topo_ruler_mm": np.nan,
        }
        fdbg = frame_debug_rows[i] if i < len(frame_debug_rows) else {}
        problema_frame = "OK"
        if cand is not None:
            valid = bool(float(cand.get("confidence", 0.0)) >= float(p.valid_score_threshold))
            base.update(cand)
            base["valid"] = valid
            if np.isfinite(px_per_mm) and px_per_mm > 0 and np.isfinite(zero_y) and np.isfinite(float(cand.get("y_top_subpx", np.nan))):
                base["topo_ruler_mm"] = (zero_y - float(cand["y_top_subpx"])) / px_per_mm
            if not valid:
                problema_frame = "Candidato encontrado, mas a confiança ficou abaixo do mínimo configurado."
        else:
            if int(fdbg.get("componentes_encontrados_na_mascara", 0)) == 0:
                problema_frame = "Nenhum objeto escuro foi encontrado na máscara da bolinha. Verifique iluminação ou threshold."
            elif int(fdbg.get("candidatos_aceitos", 0)) == 0:
                problema_frame = "A máscara encontrou objetos escuros, mas todos foram rejeitados pelos filtros de tamanho, altura, formato ou medição do topo."
            else:
                problema_frame = "Havia candidatos, mas a associação temporal não selecionou nenhum para este frame."
        base["problema_frame"] = problema_frame
        base["componentes_encontrados_na_mascara"] = int(fdbg.get("componentes_encontrados_na_mascara", 0))
        base["componentes_rejeitados"] = int(fdbg.get("componentes_rejeitados", 0))
        base["candidatos_aceitos_no_frame"] = int(fdbg.get("candidatos_aceitos", 0))
        base["threshold_usado"] = float(fdbg.get("threshold_usado", np.nan))
        base["percentil_escuro_q"] = float(fdbg.get("percentil_escuro_q", np.nan))
        rows.append(base)

        selected = bool(cand is not None)
        decision_rows.append({
            "frame": frame,
            "decisao": "candidato_selecionado" if selected else "sem_candidato_valido",
            "explicacao": problema_frame,
            "confianca_selecionada": float(base.get("confidence", 0.0)),
            "topo_ruler_mm": base.get("topo_ruler_mm", np.nan),
            "bbox_altura_px": base.get("bbox_h_px", np.nan),
            "bbox_largura_px": base.get("bbox_w_px", np.nan),
            "bbox_aspect": base.get("bbox_aspect", np.nan),
            "area_px": base.get("area_px", np.nan),
            "componentes_encontrados_na_mascara": base["componentes_encontrados_na_mascara"],
            "componentes_rejeitados": base["componentes_rejeitados"],
            "candidatos_aceitos_no_frame": base["candidatos_aceitos_no_frame"],
            "threshold_usado": base["threshold_usado"],
            "percentil_escuro_q": base["percentil_escuro_q"],
        })
        if problema_frame != "OK":
            error_lines.append(f"Frame {frame} - {problema_frame}")

    df = pd.DataFrame(rows)

    # Suavizacao mediana somente em colunas extras, sem substituir bruto.
    win = int(p.top_smooth_window)
    if win > 1:
        if win % 2 == 0:
            win += 1
        for col in ["y_top_subpx", "topo_ruler_mm", "x_center_px", "y_center_px"]:
            if col in df.columns:
                df[col + "_median"] = df[col].rolling(win, center=True, min_periods=1).median()

    # Colunas da regua e metadados.
    for k, v in ruler.items():
        if isinstance(v, (int, float, bool, str, np.floating)):
            df[k] = v
    df["ruler_height_mm_config"] = float(p.ruler_height_mm)
    df["video_width_px"] = W
    df["video_height_px"] = H
    df["ball_crop_ymin_ratio"] = float(getattr(p, "ball_crop_ymin_ratio", 0.0))
    df["ball_crop_ymax_ratio"] = float(getattr(p, "ball_crop_ymax_ratio", 1.0))
    df["ball_min_effective_threshold"] = float(getattr(p, "ball_min_effective_threshold", 75))
    df["reject_giant_border_blobs"] = bool(getattr(p, "reject_giant_border_blobs", True))
    df["ball_min_width_px_config"] = int(p.ball_min_width_px)
    df["ball_min_height_px_config"] = int(p.ball_min_height_px)
    df["ball_min_area_px_config"] = int(p.ball_min_area_px)
    df["ball_max_aspect_ratio_config"] = float(getattr(p, "ball_max_aspect_ratio", 2.20))

    df = add_time_columns(df, meta, sync_df, rise_frame, fall_frame, p)

    full_csv = os.path.join(output_dir, "tracking_full.csv")
    treated_csv = os.path.join(output_dir, "tracking_tratado.csv")
    candidates_csv = os.path.join(output_dir, "candidates.csv")
    rejection_csv = os.path.join(output_dir, "rejection_log.csv")
    decision_csv = os.path.join(output_dir, "tracking_decision_log.csv")
    errors_txt = os.path.join(output_dir, "tracking_errors.txt")

    df.to_csv(full_csv, index=False)
    if p.sync_crop_to_rising_edge and rise_frame >= 0:
        treated = df.loc[df["frame"] >= rise_frame].copy()
    else:
        treated = df.copy()
    treated.to_csv(treated_csv, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)
    pd.DataFrame(rejection_rows).to_csv(rejection_csv, index=False)
    pd.DataFrame(decision_rows).to_csv(decision_csv, index=False)
    with open(errors_txt, "w", encoding="utf-8") as f:
        if error_lines:
            f.write("\n".join(error_lines) + "\n")
        else:
            f.write("Nenhum erro crítico de tracking foi registrado.\n")

    make_preview(video_path, df, ruler, os.path.join(output_dir, "preview.jpg"), p)
    make_mask_preview(video_path, df, ruler, os.path.join(output_dir, "mask_preview.jpg"), p)
    make_ruler_preview(video_path, ruler, os.path.join(output_dir, "ruler_preview.jpg"))
    make_sync_preview(video_path, df, os.path.join(output_dir, "sync_preview.jpg"), rise_frame, fall_frame)
    make_plot(treated, os.path.join(output_dir, "plot.png"))

    print("MAGLEV VIDEO TRACKER V21")
    print(f"Video: {video_path}")
    print(f"Resolution: {W}x{H}")
    print(f"Frames: {n}")
    print(f"FPS playback: {float(df['fps_playback'].iloc[0]) if len(df) else float('nan'):.4f}")
    print(f"FPS capture:  {float(df['fps_capture'].iloc[0]) if len(df) else float('nan'):.4f}")
    print(f"Time source:  {str(df['metadata_time_source'].iloc[0]) if len(df) else 'unknown'}")
    print(f"Time conf:    {str(df['metadata_time_confidence'].iloc[0]) if len(df) else 'unknown'}")
    print(f"Rising frame: {rise_frame}")
    print(f"Falling frame: {fall_frame}")
    print(f"Ruler: {'OK' if ruler.get('ruler_ok') else 'CHECK'} | reason={ruler.get('ruler_reason')} | px/mm={ruler.get('px_per_mm'):.4f} | zero_y={ruler.get('ruler_zero_y_px'):.2f} | top_y={ruler.get('ruler_top_y_px'):.2f}")
    print(f"Ball crop y: {float(getattr(p, 'ball_crop_ymin_ratio', 0.0)):.3f} .. {float(getattr(p, 'ball_crop_ymax_ratio', 1.0)):.3f}")
    print(f"Valid frames full: {int(df['valid'].sum())}/{len(df)}")
    print(f"Valid frames treated: {int(treated['valid'].sum())}/{len(treated)}")
    if df['topo_ruler_mm'].notna().any():
        print(f"topo_ruler_mm range: {df['topo_ruler_mm'].min():.2f} .. {df['topo_ruler_mm'].max():.2f}")
    print(f"Output CSV treated: {treated_csv}")
    print(f"Output CSV full:    {full_csv}")
    print(f"Output mask preview: {os.path.join(output_dir, 'mask_preview.jpg')}")
    print(f"Output rejection log: {rejection_csv}")
    print(f"Output decision log: {decision_csv}")
    print(f"Output error log: {errors_txt}")
    print(f"Output folder: {output_dir}")


# ============================================================
# Previews e plot
# ============================================================

def make_preview(video_path: str, df: pd.DataFrame, ruler: dict, out_path: str, p: Params):
    meta = read_video_metadata(video_path)
    n = int(meta["frame_count"])
    if n <= 0:
        return
    idxs = np.linspace(0, max(0, n - 1), min(int(p.preview_frames), n)).astype(int)
    cap = cv2.VideoCapture(video_path)
    thumbs = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()
        row = df.loc[df["frame"] == int(idx)]
        row = row.iloc[0] if len(row) else None

        # Regua
        x = int(ruler.get("ruler_roi_x_px", 0)); y = int(ruler.get("ruler_roi_y_px", 0))
        w = int(ruler.get("ruler_roi_w_px", 0)); h = int(ruler.get("ruler_roi_h_px", 0))
        if w > 0 and h > 0:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 180, 0), 1)
        zy = int(round(float(ruler.get("ruler_zero_y_px", 0))))
        ty = int(round(float(ruler.get("ruler_top_y_px", 0))))
        cv2.line(vis, (max(0, x-20), zy), (min(vis.shape[1]-1, x+w+20), zy), (0, 255, 255), 1)
        cv2.line(vis, (max(0, x-20), ty), (min(vis.shape[1]-1, x+w+20), ty), (255, 255, 0), 1)

        # Linha(s) do crop vertical opcional da busca da bolinha.
        ymin_ratio = float(np.clip(getattr(p, "ball_crop_ymin_ratio", 0.0), 0.0, 1.0))
        ymax_ratio = float(np.clip(getattr(p, "ball_crop_ymax_ratio", 1.0), 0.0, 1.0))
        if ymax_ratio > ymin_ratio:
            y_min_crop = int(round(ymin_ratio * vis.shape[0]))
            y_max_crop = int(round(ymax_ratio * vis.shape[0]))
            if y_min_crop > 0:
                cv2.line(vis, (0, y_min_crop), (vis.shape[1]-1, y_min_crop), (255, 0, 0), 1)
            if y_max_crop < vis.shape[0]:
                cv2.line(vis, (0, y_max_crop), (vis.shape[1]-1, y_max_crop), (255, 0, 0), 1)

        if row is not None and bool(row.get("valid", False)) and np.isfinite(row.get("x_center_px", np.nan)):
            bx = int(round(row["bbox_x_px"])); by = int(round(row["bbox_y_px"]))
            bw = int(round(row["bbox_w_px"])); bh = int(round(row["bbox_h_px"]))
            cx = int(round(row["x_center_px"])); cy = int(round(row["y_center_px"]))
            yt = int(round(row["y_top_subpx"]))
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 255, 0), 1)
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            cv2.line(vis, (max(0, bx), yt), (min(vis.shape[1]-1, bx+bw), yt), (0, 165, 255), 2)
            txt = f"f={idx} topo={row['topo_ruler_mm']:.2f}mm conf={row['confidence']:.2f}"
        else:
            conf = float(row.get("confidence", 0.0)) if row is not None else 0.0
            problema = str(row.get("problema_frame", "INVALID")) if row is not None else "INVALID"
            txt = f"f={idx} INVALID conf={conf:.2f} {problema[:32]}"
        cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 1, cv2.LINE_AA)
        scale = float(p.preview_thumb_width_px) / vis.shape[1]
        thumb = cv2.resize(vis, (int(p.preview_thumb_width_px), int(vis.shape[0] * scale)))
        thumbs.append(thumb)
    cap.release()
    if not thumbs:
        return
    cols = 3
    rows = int(math.ceil(len(thumbs) / cols))
    th, tw = thumbs[0].shape[:2]
    canvas = np.full((rows * th, cols * tw, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        rr = i // cols; cc = i % cols
        canvas[rr*th:rr*th+thumb.shape[0], cc*tw:cc*tw+thumb.shape[1]] = thumb
    cv2.imwrite(out_path, canvas)




def make_mask_preview(video_path: str, df: pd.DataFrame, ruler: dict, out_path: str, p: Params):
    """Preview de depuracao da mascara escura da bolinha.

    Verde = pixels aceitos pela mascara. A linha azul mostra o crop vertical da
    bolinha. Se der INVALID, este preview mostra se a bolinha sumiu da mascara ou
    se grudou em um blob gigante do fundo/base.
    """
    meta = read_video_metadata(video_path)
    n = int(meta["frame_count"])
    if n <= 0:
        return
    idxs = np.linspace(0, max(0, n - 1), min(int(p.preview_frames), n)).astype(int)
    cap = cv2.VideoCapture(video_path)
    thumbs = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        mask = make_dark_ball_mask(frame, ruler, p)
        vis = frame.copy()
        overlay = vis.copy()
        overlay[mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(overlay, 0.38, vis, 0.62, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        allowed = exclusion_mask_for_ruler_and_led(frame, ruler, p)
        thr, qthr = compute_dark_threshold(cv2.GaussianBlur(gray, (5, 5), 0), allowed, p)

        H, W = frame.shape[:2]
        y_min_crop, y_max_crop = get_ball_crop_limits(H, p)
        if y_min_crop > 0:
            cv2.line(vis, (0, y_min_crop), (W - 1, y_min_crop), (255, 0, 0), 1)
        if y_max_crop < H:
            cv2.line(vis, (0, y_max_crop), (W - 1, y_max_crop), (255, 0, 0), 1)

        x = int(ruler.get("ruler_roi_x_px", 0)); y = int(ruler.get("ruler_roi_y_px", 0))
        w = int(ruler.get("ruler_roi_w_px", 0)); h = int(ruler.get("ruler_roi_h_px", 0))
        if w > 0 and h > 0:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 180, 0), 1)

        row = df.loc[df["frame"] == int(idx)]
        row = row.iloc[0] if len(row) else None
        if row is not None and bool(row.get("valid", False)) and np.isfinite(row.get("bbox_x_px", np.nan)):
            bx = int(round(row["bbox_x_px"])); by = int(round(row["bbox_y_px"]))
            bw = int(round(row["bbox_w_px"])); bh = int(round(row["bbox_h_px"]))
            yt = int(round(row["y_top_subpx"]))
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
            cv2.line(vis, (max(0, bx), yt), (min(W - 1, bx + bw), yt), (0, 165, 255), 2)
            status = f"VALID topo={row['topo_ruler_mm']:.2f}mm conf={row['confidence']:.2f}"
        else:
            problema = str(row.get("problema_frame", "INVALID")) if row is not None else "INVALID"
            status = f"INVALID {problema[:28]}"
        txt = f"f={idx} {status} thr={thr:.1f} q={qthr:.1f}"
        cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255,255,255), 1, cv2.LINE_AA)
        scale = float(p.preview_thumb_width_px) / vis.shape[1]
        thumbs.append(cv2.resize(vis, (int(p.preview_thumb_width_px), int(vis.shape[0] * scale))))
    cap.release()
    if not thumbs:
        return
    cols = 3
    rows = int(math.ceil(len(thumbs) / cols))
    th, tw = thumbs[0].shape[:2]
    canvas = np.full((rows * th, cols * tw, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        rr = i // cols; cc = i % cols
        canvas[rr*th:rr*th+thumb.shape[0], cc*tw:cc*tw+thumb.shape[1]] = thumb
    cv2.imwrite(out_path, canvas)


def make_ruler_preview(video_path: str, ruler: dict, out_path: str):
    frame = sample_frame(video_path, 0)
    vis = frame.copy()
    x = int(ruler.get("ruler_roi_x_px", 0)); y = int(ruler.get("ruler_roi_y_px", 0))
    w = int(ruler.get("ruler_roi_w_px", 0)); h = int(ruler.get("ruler_roi_h_px", 0))
    if w > 0 and h > 0:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 180, 0), 2)
    zero_y = int(round(float(ruler.get("ruler_zero_y_px", 0))))
    top_y = int(round(float(ruler.get("ruler_top_y_px", 0))))
    cv2.line(vis, (0, zero_y), (vis.shape[1]-1, zero_y), (0, 255, 255), 2)
    cv2.line(vis, (0, top_y), (vis.shape[1]-1, top_y), (255, 255, 0), 2)
    txt = f"ruler {'OK' if ruler.get('ruler_ok') else 'CHECK'} | {ruler.get('px_per_mm', float('nan')):.2f} px/mm | H={ruler.get('ruler_height_mm', float('nan')):.1f} mm"
    cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def make_sync_preview(video_path: str, df: pd.DataFrame, out_path: str, rise_frame: int, fall_frame: int):
    if rise_frame < 0 and fall_frame < 0:
        return
    meta = read_video_metadata(video_path)
    n = int(meta["frame_count"])
    frames = []
    for f in [rise_frame-2, rise_frame-1, rise_frame, rise_frame+1, fall_frame-1, fall_frame, fall_frame+1]:
        if 0 <= f < n and f not in frames:
            frames.append(f)
    cap = cv2.VideoCapture(video_path)
    thumbs = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()
        row = df.loc[df["frame"] == int(f)]
        if len(row):
            r = row.iloc[0]
            if np.isfinite(r.get("sync_red_centroid_x_px", np.nan)):
                cx = int(round(r["sync_red_centroid_x_px"])); cy = int(round(r["sync_red_centroid_y_px"]))
                cv2.circle(vis, (cx, cy), 18, (0, 255, 255), 2)
            txt = f"f={f} {r.get('sync_event','')} score={float(r.get('sync_score',0)):.2f}"
        else:
            txt = f"f={f}"
        cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,255,255), 1, cv2.LINE_AA)
        scale = 420 / vis.shape[1]
        thumbs.append(cv2.resize(vis, (420, int(vis.shape[0] * scale))))
    cap.release()
    if not thumbs:
        return
    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    th, tw = thumbs[0].shape[:2]
    canvas = np.full((rows * th, cols * tw, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        rr = i // cols; cc = i % cols
        canvas[rr*th:rr*th+thumb.shape[0], cc*tw:cc*tw+thumb.shape[1]] = thumb
    cv2.imwrite(out_path, canvas)


def make_plot(df: pd.DataFrame, out_path: str):
    import matplotlib.pyplot as plt
    if df.empty:
        return
    time_conf = str(df["metadata_time_confidence"].iloc[0]) if "metadata_time_confidence" in df.columns else ""
    unsafe_time = time_conf in {"LOW", "LOW_SYNC_WIDTH"}
    if (not unsafe_time) and "time_sync_ms" in df.columns and df["time_sync_ms"].notna().any():
        x = df["time_sync_ms"]
        xlabel = "time_sync_ms (ms)"
    elif "frame_sync" in df.columns and df["frame_sync"].notna().any():
        x = df["frame_sync"]
        xlabel = "frame_sync (frames)"
    else:
        x = df["frame"]
        xlabel = "frame"

    valid = df["valid"].astype(bool) if "valid" in df.columns else pd.Series(True, index=df.index)
    fig = plt.figure(figsize=(16, 7))
    ax = plt.gca()
    if "topo_ruler_mm_median" in df.columns and df["topo_ruler_mm_median"].notna().any():
        ax.plot(x, df["topo_ruler_mm_median"].where(valid), label="topo_ruler_mm median", linewidth=1.7)
    if "topo_ruler_mm" in df.columns and df["topo_ruler_mm"].notna().any():
        ax.plot(x, df["topo_ruler_mm"].where(valid), label="topo_ruler_mm raw", linewidth=0.8, alpha=0.55)
    if "sync_rising_edge" in df.columns:
        if xlabel.startswith("time_sync"):
            ax.axvline(0.0, linestyle="--", linewidth=1.0, label="RISING_EDGE")
        else:
            rr = df.index[df["sync_rising_edge"].fillna(0).astype(int) == 1]
            if len(rr):
                ax.axvline(float(x.loc[rr[0]]), linestyle="--", linewidth=1.0, label="RISING_EDGE")
    if "sync_falling_edge" in df.columns:
        ff = df.index[df["sync_falling_edge"].fillna(0).astype(int) == 1]
        if len(ff):
            ax.axvline(float(x.loc[ff[0]]), linestyle=":", linewidth=1.0, label="FALLING_EDGE")
    ax.set_title(f"Maglev video tracking V21 | time_conf={time_conf}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("topo da bolinha na regua (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def load_config(path: str) -> Params:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("//"):
            continue
        lines.append(line)
    data = json.loads("\n".join(lines))
    p = Params()
    for k, v in data.items():
        if hasattr(p, k):
            setattr(p, k, v)
    if isinstance(p.manual_ruler_roi, list):
        p.manual_ruler_roi = tuple(int(x) for x in p.manual_ruler_roi)  # type: ignore
    return p


def save_default_config(path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(Params()), f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Caminho do video de entrada")
    ap.add_argument("--output-dir", default=None, help="Pasta de saida. Default: nome do video + _v21")
    ap.add_argument("--config", default=None, help="JSON opcional de parametros")
    ap.add_argument("--write-default-config", default=None, help="Escreve config default e sai")

    ap.add_argument("--ruler-height-mm", type=float, default=None, help="Altura fisica da regua visivel. Default: 26 mm")
    ap.add_argument("--ruler-roi", default=None, help="Fallback manual opcional x,y,w,h da regua")
    ap.add_argument("--ruler-top-y-px", type=float, default=None, help="Fallback manual: y do topo da regua")
    ap.add_argument("--ruler-zero-y-px", type=float, default=None, help="Fallback manual: y do zero da regua")
    ap.add_argument("--ruler-search-x0-ratio", type=float, default=None, help="Inicio da busca da regua no eixo x. Default: 0.50")

    ap.add_argument("--ball-dark-threshold", type=int, default=None, help="Threshold absoluto maximo para bolinha escura. Default: 120")
    ap.add_argument("--min-dark-threshold", type=int, default=None, help="Piso do threshold efetivo da mascara escura. Default: 75")
    ap.add_argument("--ball-dark-percentile", type=float, default=None, help="Percentil escuro do frame. Default: 22")
    ap.add_argument("--ball-min-area-px", type=int, default=None, help="Area minima do blob da bolinha")
    ap.add_argument("--ball-min-width-px", type=int, default=None, help="Largura minima do objeto para aceitar como bolinha. Default: 80")
    ap.add_argument("--ball-min-height-px", type=int, default=None, help="Altura minima do objeto para aceitar como bolinha. Default: 70")
    ap.add_argument("--ball-max-aspect-ratio", type=float, default=None, help="Maximo largura/altura aceito. Rejeita fendas horizontais. Default: 2.2")
    ap.add_argument("--ball-crop-ymax-ratio", type=float, default=None, help="Alias legado. Use --max-y-crop.")
    ap.add_argument("--max-y-crop", type=float, default=None, help="Crop vertical maximo da bolinha. Ex: 0.65 mantem 65%% superiores do frame")
    ap.add_argument("--ball-crop-ymin-ratio", type=float, default=None, help="Alias legado. Use --min-y-crop.")
    ap.add_argument("--min-y-crop", type=float, default=None, help="Crop vertical minimo da bolinha. Default: 0.0")
    ap.add_argument("--top-percentile", type=float, default=None, help="Percentil usado para topo do blob. Default: 6")
    ap.add_argument("--top-smooth-window", type=int, default=None, help="Janela mediana do topo. Default: 7")

    ap.add_argument("--no-sync-crop", action="store_true", help="Nao corta CSV tratado no rising edge")
    ap.add_argument("--sync-known-pulse-width-ms", type=float, default=None, help="Duracao conhecida do LED para fallback temporal")

    args = ap.parse_args()

    if args.write_default_config:
        save_default_config(args.write_default_config)
        print(f"Config default salva em: {args.write_default_config}")
        return

    p = load_config(args.config) if args.config else Params()
    if args.ruler_height_mm is not None:
        p.ruler_height_mm = float(args.ruler_height_mm)
    if args.ruler_roi:
        p.manual_ruler_roi = parse_tuple4(args.ruler_roi)
    if args.ruler_top_y_px is not None:
        p.manual_ruler_top_y_px = float(args.ruler_top_y_px)
    if args.ruler_zero_y_px is not None:
        p.manual_ruler_zero_y_px = float(args.ruler_zero_y_px)
    if args.ruler_search_x0_ratio is not None:
        p.ruler_search_x0_ratio = float(args.ruler_search_x0_ratio)
    if args.ball_dark_threshold is not None:
        p.ball_dark_threshold = int(args.ball_dark_threshold)
    if args.min_dark_threshold is not None:
        p.ball_min_effective_threshold = int(args.min_dark_threshold)
    if args.ball_dark_percentile is not None:
        p.ball_dark_percentile = float(args.ball_dark_percentile)
    if args.ball_min_area_px is not None:
        p.ball_min_area_px = int(args.ball_min_area_px)
    if args.ball_min_width_px is not None:
        p.ball_min_width_px = int(args.ball_min_width_px)
    if args.ball_min_height_px is not None:
        p.ball_min_height_px = int(args.ball_min_height_px)
    if args.ball_max_aspect_ratio is not None:
        p.ball_max_aspect_ratio = float(args.ball_max_aspect_ratio)
    if args.ball_crop_ymax_ratio is not None:
        p.ball_crop_ymax_ratio = float(args.ball_crop_ymax_ratio)
    if args.max_y_crop is not None:
        p.ball_crop_ymax_ratio = float(args.max_y_crop)
    if args.ball_crop_ymin_ratio is not None:
        p.ball_crop_ymin_ratio = float(args.ball_crop_ymin_ratio)
    if args.min_y_crop is not None:
        p.ball_crop_ymin_ratio = float(args.min_y_crop)
    if args.top_percentile is not None:
        p.top_percentile = float(args.top_percentile)
    if args.top_smooth_window is not None:
        p.top_smooth_window = int(args.top_smooth_window)
    if args.no_sync_crop:
        p.sync_crop_to_rising_edge = False
    if args.sync_known_pulse_width_ms is not None:
        p.sync_known_pulse_width_ms = float(args.sync_known_pulse_width_ms)

    stem = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = args.output_dir if args.output_dir else f"{stem}_v21"
    process_video(args.video, output_dir, p)


if __name__ == "__main__":
    main()
