
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAGLEV VIDEO TRACKER V6 - AUTO ROI + FIXED RADIUS + RULER CALIBRATION

Objetivo:
    Ler um vídeo do Maglev e gerar um CSV com a posição da bolinha ao longo do tempo.

Saída principal:
    frame
    time_video_s
    time_real_s
    x_center_px
    y_center_px
    radius_px
    x_left_px
    y_top_px
    x_right_px
    y_bottom_px
    confidence
    valid
    method

Ideia do modelo:
    1) Estima background/frames de referência.
    2) Detecta automaticamente uma ROI de movimento.
    3) Dentro da ROI, procura candidatos de bolinha por:
        - segmentação de objeto claro/branco;
        - diferença em relação ao background;
        - geometria circular;
        - estabilidade temporal.
    4) Escolhe a trajetória por programação dinâmica simples,
       penalizando saltos impossíveis.
    5) Exporta CSV e imagens de preview/debug.

Observação importante:
    Este script NÃO usa Kalman como detector.
    Ele usa visão computacional clássica e só aplica suavização opcional depois.

Dependências:
    pip install opencv-python numpy pandas matplotlib

Exemplo:
    python maglev_video_tracker_v4_auto.py --video ident_slow_mo.mp4
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd


@dataclass
class Params:
    # -------------------------
    # Tempo
    # -------------------------
    time_scale: float = 1.0
    # time_real_s = time_video_s * time_scale
    # Se o vídeo foi gravado em 120 fps e salvo/reproduzido em 30 fps:
    # time_scale = 30/120 = 0.25

    # -------------------------
    # Raio fixo / metrologia
    # -------------------------
    # Se fixed_radius_px > 0, o topo físico passa a ser:
    #     y_top_fixed_px = y_center_px - fixed_radius_px
    # Isso reduz muito ruído de contorno/reflexo.
    fixed_radius_px: float = 0.0

    # Alternativa: informar o diâmetro real e a escala do vídeo.
    # Se ball_diameter_mm > 0 e px_per_mm > 0:
    #     fixed_radius_px = (ball_diameter_mm / 2) * px_per_mm
    ball_diameter_mm: float = 0.0
    px_per_mm: float = 0.0
    mm_per_px: float = 0.0

    # Calibração pela régua visível no vídeo.
    # Exemplo: se a marca 15 cm está em y=80 px e a 17 cm em y=250 px:
    #     ruler_y_top_px = 80
    #     ruler_y_bottom_px = 250
    #     ruler_top_cm = 15
    #     ruler_bottom_cm = 17
    #
    # O script calcula:
    #     mm_per_px = ((17-15)*10) / (250-80)
    #
    # Como y cresce para baixo, isso mapeia a posição vertical para a coordenada real da régua.
    ruler_y_top_px: float = 0.0
    ruler_y_bottom_px: float = 0.0
    ruler_top_cm: float = 15.0
    ruler_bottom_cm: float = 17.0

    # Se fixed_radius_px == 0 e auto_fixed_radius=True,
    # usa a mediana do radius_px detectado nos frames válidos.
    auto_fixed_radius: bool = True
    radius_median_min_confidence: float = 0.55

    # -------------------------
    # ROI automática
    # -------------------------
    auto_roi: bool = True
    roi: Optional[Tuple[int, int, int, int]] = None
    roi_margin_px: int = 35
    motion_sample_count: int = 80
    motion_threshold_percentile: float = 97.5
    motion_min_area_px: int = 80

    # Restringe a ROI final para evitar bordas e UI do vídeo
    frame_border_ignore_px: int = 4

    # -------------------------
    # Segmentação da bolinha
    # -------------------------
    # A bolinha no vídeo parece clara/metálica. Estes pesos priorizam regiões claras,
    # mas exigem movimento/diferença de background.
    min_ball_radius_px: int = 18
    max_ball_radius_px: int = 85

    bright_percentile: float = 86.0
    diff_percentile: float = 88.0
    min_contour_area_px: int = 350
    max_contour_area_px: int = 16000

    # Critérios geométricos
    min_circularity: float = 0.35
    min_fill_ratio: float = 0.25
    max_fill_ratio: float = 1.35
    max_aspect_deviation: float = 0.65
    # aspect_deviation = abs(w-h)/max(w,h)

    # -------------------------
    # Fusão / pontuação
    # -------------------------
    # Score candidato = combinação ponderada.
    w_brightness: float = 0.22
    w_motion: float = 0.30
    w_circularity: float = 0.20
    w_size: float = 0.12
    w_roi_center_prior: float = 0.04
    w_template: float = 0.12

    # Template opcional, atualizado a partir do melhor candidato do primeiro trecho.
    # Use 0 se quiser desligar.
    template_update_alpha: float = 0.03
    template_min_confidence_to_update: float = 0.70

    # -------------------------
    # Associação temporal
    # -------------------------
    # Programação dinâmica: favorece continuidade, mas permite saltos reais.
    max_jump_px_soft: float = 55.0
    jump_penalty_weight: float = 0.010
    missing_penalty: float = 0.80

    # Se o objeto for perdido, ainda assim escolhe o melhor candidato local/global.
    max_candidates_per_frame: int = 8

    # -------------------------
    # Validade / confiança
    # -------------------------
    valid_confidence_threshold: float = 0.38

    # -------------------------
    # Pós-processamento opcional
    # -------------------------
    # Não altera y_center_px/y_top_px bruto. Cria colunas suavizadas se > 1.
    median_smooth_window: int = 1

    # -------------------------
    # Preview
    # -------------------------
    preview_frames: int = 18
    preview_width_px: int = 1800


def parse_tuple4(s: str) -> Tuple[int, int, int, int]:
    vals = [int(float(v.strip())) for v in s.split(",")]
    if len(vals) != 4:
        raise ValueError("Esperado formato x,y,w,h")
    return tuple(vals)  # type: ignore


def clamp_roi(roi: Tuple[int, int, int, int], W: int, H: int, border: int = 0) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(border, min(x, W - 1 - border))
    y = max(border, min(y, H - 1 - border))
    x2 = max(x + 1, min(x + w, W - border))
    y2 = max(y + 1, min(y + h, H - border))
    return int(x), int(y), int(x2 - x), int(y2 - y)


def read_video_metadata(video_path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir o vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": float(fps), "frame_count": int(n), "width": W, "height": H, "duration_s": float(n / fps if fps else 0)}


def sample_frames(video_path: str, count: int = 80) -> List[Tuple[int, np.ndarray]]:
    meta = read_video_metadata(video_path)
    n = int(meta["frame_count"])
    if n <= 0:
        return []
    idxs = np.linspace(0, n - 1, min(count, n)).astype(int)
    cap = cv2.VideoCapture(video_path)
    out = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            out.append((int(idx), frame))
    cap.release()
    return out


def estimate_background(frames: List[Tuple[int, np.ndarray]]) -> np.ndarray:
    if not frames:
        raise RuntimeError("Sem frames para background")
    arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in frames], axis=0)
    bg = np.median(arr, axis=0).astype(np.uint8)
    return bg


def auto_motion_roi(video_path: str, p: Params) -> Tuple[int, int, int, int]:
    samples = sample_frames(video_path, p.motion_sample_count)
    if not samples:
        raise RuntimeError("Não consegui amostrar frames")
    H, W = samples[0][1].shape[:2]
    bg = estimate_background(samples)

    acc = np.zeros((H, W), dtype=np.float32)
    for _, frame in samples:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, bg)
        acc += diff.astype(np.float32)
    acc /= max(1, len(samples))

    # ignora bordas
    b = p.frame_border_ignore_px
    if b > 0:
        acc[:b, :] = 0
        acc[-b:, :] = 0
        acc[:, :b] = 0
        acc[:, -b:] = 0

    thr = np.percentile(acc[acc > 0], p.motion_threshold_percentile) if np.any(acc > 0) else 255
    mask = (acc >= thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= p.motion_min_area_px:
            boxes.append(cv2.boundingRect(c))

    if not boxes:
        # fallback: região central-direita, onde estava a bolinha nos vídeos enviados
        return clamp_roi((int(W * 0.45), int(H * 0.05), int(W * 0.38), int(H * 0.85)), W, H, p.frame_border_ignore_px)

    x1 = min(x for x, y, w, h in boxes)
    y1 = min(y for x, y, w, h in boxes)
    x2 = max(x + w for x, y, w, h in boxes)
    y2 = max(y + h for x, y, w, h in boxes)

    m = p.roi_margin_px
    roi = (x1 - m, y1 - m, (x2 - x1) + 2 * m, (y2 - y1) + 2 * m)
    return clamp_roi(roi, W, H, p.frame_border_ignore_px)


def normalize01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def circularity_of_contour(c) -> float:
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri <= 1e-9:
        return 0.0
    return float(4 * math.pi * area / (peri * peri))


def candidate_from_contour(
    c,
    roi_offset: Tuple[int, int],
    gray_roi: np.ndarray,
    diff_roi: np.ndarray,
    template: Optional[np.ndarray],
    p: Params,
) -> Optional[dict]:
    area = cv2.contourArea(c)
    if area < p.min_contour_area_px or area > p.max_contour_area_px:
        return None

    x, y, w, h = cv2.boundingRect(c)
    if w <= 2 or h <= 2:
        return None

    radius = 0.5 * max(w, h)
    if radius < p.min_ball_radius_px or radius > p.max_ball_radius_px:
        return None

    aspect_dev = abs(w - h) / max(w, h)
    if aspect_dev > p.max_aspect_deviation:
        return None

    circ = circularity_of_contour(c)
    if circ < p.min_circularity:
        return None

    fill = area / max(1.0, float(w * h))
    if not (p.min_fill_ratio <= fill <= p.max_fill_ratio):
        return None

    mask = np.zeros(gray_roi.shape, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    mean_bright = float(cv2.mean(gray_roi, mask=mask)[0])
    mean_diff = float(cv2.mean(diff_roi, mask=mask)[0])

    # Centro por momentos; fallback no bbox
    M = cv2.moments(c)
    if abs(M["m00"]) > 1e-9:
        cx_roi = float(M["m10"] / M["m00"])
        cy_roi = float(M["m01"] / M["m00"])
    else:
        cx_roi = x + w / 2
        cy_roi = y + h / 2

    ox, oy = roi_offset
    cx = cx_roi + ox
    cy = cy_roi + oy

    # Priors normalizados
    bright_score = mean_bright / 255.0
    motion_score = min(1.0, mean_diff / 80.0)
    circ_score = min(1.0, circ)
    # size score com pico no meio da faixa
    rmid = 0.5 * (p.min_ball_radius_px + p.max_ball_radius_px)
    rspan = 0.5 * (p.max_ball_radius_px - p.min_ball_radius_px)
    size_score = 1.0 - min(1.0, abs(radius - rmid) / max(1.0, rspan))

    # prior fraco: objeto tende a ficar dentro da ROI, mas não força centro
    H, W = gray_roi.shape[:2]
    dx = abs(cx_roi - W / 2) / max(1, W / 2)
    dy = abs(cy_roi - H / 2) / max(1, H / 2)
    roi_center_prior = 1.0 - min(1.0, math.sqrt(dx * dx + dy * dy) / math.sqrt(2))

    template_score = 0.0
    if template is not None:
        # compara crop redimensionado com template
        crop = gray_roi[max(0, y):min(gray_roi.shape[0], y + h), max(0, x):min(gray_roi.shape[1], x + w)]
        if crop.size > 0 and template.size > 0:
            try:
                crop_resized = cv2.resize(crop, (template.shape[1], template.shape[0]))
                res = cv2.matchTemplate(crop_resized, template, cv2.TM_CCOEFF_NORMED)
                template_score = float(res[0, 0])
                template_score = (template_score + 1) / 2  # -1..1 para 0..1
            except Exception:
                template_score = 0.0

    score = (
        p.w_brightness * bright_score
        + p.w_motion * motion_score
        + p.w_circularity * circ_score
        + p.w_size * size_score
        + p.w_roi_center_prior * roi_center_prior
        + p.w_template * template_score
    )

    # raio equivalente por área e raio bbox; usa média robusta
    r_area = math.sqrt(max(area, 1.0) / math.pi)
    r_bbox = 0.25 * (w + h)
    r = 0.55 * r_area + 0.45 * r_bbox

    return {
        "x_center_px": cx,
        "y_center_px": cy,
        "radius_px": r,
        "x_left_px": cx - r,
        "y_top_px": cy - r,
        "x_right_px": cx + r,
        "y_bottom_px": cy + r,
        "confidence": float(score),
        "method": "contour",
        "area": float(area),
        "circularity": float(circ),
        "fill_ratio": float(fill),
        "brightness": mean_bright,
        "motion": mean_diff,
        "template_score": float(template_score),
    }


def detect_candidates_in_frame(frame: np.ndarray, bg_gray: np.ndarray, roi: Tuple[int, int, int, int], template: Optional[np.ndarray], p: Params) -> List[dict]:
    x0, y0, w0, h0 = roi
    frame_roi = frame[y0:y0+h0, x0:x0+w0]
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    bg_roi = bg_gray[y0:y0+h0, x0:x0+w0]
    diff = cv2.absdiff(gray, bg_roi)

    # equalização leve para reduzir variação de luz
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)

    bright_thr = np.percentile(gray_blur, p.bright_percentile)
    diff_thr = np.percentile(diff_blur, p.diff_percentile)

    # máscara: claro OU diferença forte, mas com preferência para regiões com diferença
    bright_mask = (gray_blur >= bright_thr).astype(np.uint8) * 255
    diff_mask = (diff_blur >= diff_thr).astype(np.uint8) * 255

    # combina; a bola metálica pode ser clara e móvel
    mask = cv2.bitwise_and(bright_mask, cv2.dilate(diff_mask, None, iterations=1))
    # fallback: se ficou vazio, usa diferença
    if cv2.countNonZero(mask) < 50:
        mask = diff_mask.copy()

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        cand = candidate_from_contour(c, (x0, y0), gray, diff_blur, template, p)
        if cand is not None:
            candidates.append(cand)

    # Hough como complemento, mas sem dominar
    # Útil quando contorno fragmenta.
    try:
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, p.min_ball_radius_px),
            param1=80,
            param2=18,
            minRadius=p.min_ball_radius_px,
            maxRadius=p.max_ball_radius_px,
        )
        if circles is not None:
            for c in np.round(circles[0, :]).astype(int)[:10]:
                cxr, cyr, rr = int(c[0]), int(c[1]), int(c[2])
                if rr < p.min_ball_radius_px or rr > p.max_ball_radius_px:
                    continue
                x1 = max(0, cxr - rr)
                y1 = max(0, cyr - rr)
                x2 = min(gray.shape[1], cxr + rr)
                y2 = min(gray.shape[0], cyr + rr)
                patch_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(patch_mask, (cxr, cyr), rr, 255, -1)
                mean_bright = float(cv2.mean(gray, mask=patch_mask)[0])
                mean_diff = float(cv2.mean(diff_blur, mask=patch_mask)[0])
                bright_score = mean_bright / 255.0
                motion_score = min(1.0, mean_diff / 80.0)
                size_score = 1.0 - min(1.0, abs(rr - 0.5*(p.min_ball_radius_px+p.max_ball_radius_px)) / max(1.0, 0.5*(p.max_ball_radius_px-p.min_ball_radius_px)))
                score = 0.30*motion_score + 0.25*bright_score + 0.25*size_score + 0.20
                candidates.append({
                    "x_center_px": float(cxr + x0),
                    "y_center_px": float(cyr + y0),
                    "radius_px": float(rr),
                    "x_left_px": float(cxr + x0 - rr),
                    "y_top_px": float(cyr + y0 - rr),
                    "x_right_px": float(cxr + x0 + rr),
                    "y_bottom_px": float(cyr + y0 + rr),
                    "confidence": float(score),
                    "method": "hough",
                    "area": float(math.pi*rr*rr),
                    "circularity": 1.0,
                    "fill_ratio": 0.78,
                    "brightness": mean_bright,
                    "motion": mean_diff,
                    "template_score": 0.0,
                })
    except Exception:
        pass

    # Deduplicação por proximidade
    candidates.sort(key=lambda d: d["confidence"], reverse=True)
    dedup = []
    for c in candidates:
        keep = True
        for d in dedup:
            dist = math.hypot(c["x_center_px"] - d["x_center_px"], c["y_center_px"] - d["y_center_px"])
            if dist < 0.45 * (c["radius_px"] + d["radius_px"]):
                keep = False
                break
        if keep:
            dedup.append(c)
        if len(dedup) >= p.max_candidates_per_frame:
            break

    return dedup


def choose_track_dp(all_candidates: List[List[dict]], p: Params) -> List[Optional[dict]]:
    """
    Escolhe uma trajetória usando programação dinâmica.
    Penaliza baixa confiança e saltos grandes entre frames.
    """
    n = len(all_candidates)
    if n == 0:
        return []

    # cada frame pode ter candidatos; se vazio, usa None
    costs = []
    prevs = []

    for i, cands in enumerate(all_candidates):
        if not cands:
            cands = [None]
        frame_costs = []
        frame_prevs = []

        for j, cand in enumerate(cands):
            det_cost = p.missing_penalty if cand is None else (1.0 - float(cand["confidence"]))
            if i == 0:
                frame_costs.append(det_cost)
                frame_prevs.append(-1)
            else:
                best = float("inf")
                best_k = -1
                prev_cands = all_candidates[i-1] if all_candidates[i-1] else [None]
                for k, pcand in enumerate(prev_cands):
                    prev_cost = costs[i-1][k]
                    jump_cost = 0.0
                    if cand is None or pcand is None:
                        jump_cost = 0.15
                    else:
                        dx = cand["x_center_px"] - pcand["x_center_px"]
                        dy = cand["y_center_px"] - pcand["y_center_px"]
                        dist = math.hypot(dx, dy)
                        # Huber-like: permite saltos reais, penaliza extremos
                        excess = max(0.0, dist - p.max_jump_px_soft)
                        jump_cost = p.jump_penalty_weight * (dist + 2.5 * excess)
                    total = prev_cost + det_cost + jump_cost
                    if total < best:
                        best = total
                        best_k = k
                frame_costs.append(best)
                frame_prevs.append(best_k)

        costs.append(frame_costs)
        prevs.append(frame_prevs)

    # backtrack
    last_idx = int(np.argmin(costs[-1]))
    track = [None] * n
    idx = last_idx
    for i in range(n - 1, -1, -1):
        cands = all_candidates[i] if all_candidates[i] else [None]
        track[i] = cands[idx]
        idx = prevs[i][idx]
        if idx < 0 and i > 0:
            idx = 0
    return track


def maybe_update_template(frame: np.ndarray, cand: dict, template: Optional[np.ndarray], p: Params) -> Optional[np.ndarray]:
    if cand is None or cand.get("confidence", 0) < p.template_min_confidence_to_update:
        return template
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x1 = int(round(cand["x_left_px"]))
    y1 = int(round(cand["y_top_px"]))
    x2 = int(round(cand["x_right_px"]))
    y2 = int(round(cand["y_bottom_px"]))
    H, W = gray.shape
    x1 = max(0, min(W-1, x1))
    x2 = max(x1+1, min(W, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(y1+1, min(H, y2))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return template

    if template is None:
        return cv2.resize(crop, (64, 64))
    crop = cv2.resize(crop, (template.shape[1], template.shape[0]))
    return cv2.addWeighted(template, 1.0 - p.template_update_alpha, crop, p.template_update_alpha, 0)


def process_video(video_path: str, out_csv: str, p: Params, preview: Optional[str] = None, plot: Optional[str] = None, debug_candidates: Optional[str] = None):
    meta = read_video_metadata(video_path)
    fps = meta["fps"]
    W, H = int(meta["width"]), int(meta["height"])
    n = int(meta["frame_count"])

    samples = sample_frames(video_path, p.motion_sample_count)
    bg_gray = estimate_background(samples)

    if p.auto_roi or p.roi is None:
        roi = auto_motion_roi(video_path, p)
    else:
        roi = clamp_roi(p.roi, W, H, p.frame_border_ignore_px)

    cap = cv2.VideoCapture(video_path)
    all_candidates: List[List[dict]] = []
    frames_for_template: List[np.ndarray] = []
    candidate_rows = []

    template = None

    # Primeiro passe: detecta candidatos
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cands = detect_candidates_in_frame(frame, bg_gray, roi, template, p)

        # Atualiza template com o melhor candidato bruto, com muita cautela.
        if cands:
            template = maybe_update_template(frame, cands[0], template, p)

        all_candidates.append(cands)

        if debug_candidates:
            for rank, c in enumerate(cands):
                row = {"frame": frame_idx, "rank": rank}
                row.update(c)
                candidate_rows.append(row)

        frame_idx += 1

    cap.release()

    track = choose_track_dp(all_candidates, p)

    rows = []
    for i, cand in enumerate(track):
        time_video_s = i / fps if fps else np.nan
        time_real_s = time_video_s * p.time_scale
        if cand is None:
            rows.append({
                "frame": i,
                "time_video_s": time_video_s,
                "time_real_s": time_real_s,
                "x_center_px": np.nan,
                "y_center_px": np.nan,
                "radius_px": np.nan,
                "x_left_px": np.nan,
                "y_top_px": np.nan,
                "x_right_px": np.nan,
                "y_bottom_px": np.nan,
                "confidence": 0.0,
                "valid": False,
                "method": "missing",
            })
        else:
            rows.append({
                "frame": i,
                "time_video_s": time_video_s,
                "time_real_s": time_real_s,
                "x_center_px": cand["x_center_px"],
                "y_center_px": cand["y_center_px"],
                "radius_px": cand["radius_px"],
                "x_left_px": cand["x_left_px"],
                "y_top_px": cand["y_top_px"],
                "x_right_px": cand["x_right_px"],
                "y_bottom_px": cand["y_bottom_px"],
                "confidence": cand["confidence"],
                "valid": bool(cand["confidence"] >= p.valid_confidence_threshold),
                "method": cand.get("method", "unknown"),
            })

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # V5: raio fixo para reduzir ruído do topo
    # ------------------------------------------------------------
    fixed_radius_px = float(p.fixed_radius_px or 0.0)

    if fixed_radius_px <= 0 and p.ball_diameter_mm and p.ball_diameter_mm > 0:
        if p.px_per_mm and p.px_per_mm > 0:
            fixed_radius_px = 0.5 * float(p.ball_diameter_mm) * float(p.px_per_mm)
        elif p.mm_per_px and p.mm_per_px > 0:
            fixed_radius_px = 0.5 * float(p.ball_diameter_mm) / float(p.mm_per_px)

    if fixed_radius_px <= 0 and p.auto_fixed_radius:
        good_radius = df.loc[
            (df["confidence"] >= p.radius_median_min_confidence) & df["radius_px"].notna(),
            "radius_px",
        ]
        if len(good_radius) >= 5:
            fixed_radius_px = float(good_radius.median())
        elif df["radius_px"].notna().any():
            fixed_radius_px = float(df["radius_px"].median())

    df["fixed_radius_px"] = fixed_radius_px if fixed_radius_px > 0 else np.nan

    if fixed_radius_px > 0:
        df["y_top_fixed_px"] = df["y_center_px"] - fixed_radius_px
        df["y_bottom_fixed_px"] = df["y_center_px"] + fixed_radius_px
        df["x_left_fixed_px"] = df["x_center_px"] - fixed_radius_px
        df["x_right_fixed_px"] = df["x_center_px"] + fixed_radius_px
    else:
        df["y_top_fixed_px"] = np.nan
        df["y_bottom_fixed_px"] = np.nan
        df["x_left_fixed_px"] = np.nan
        df["x_right_fixed_px"] = np.nan

    # Conversão opcional para mm.
    # Prioridade:
    #   1) calibração pela régua: y_top_px/y_bottom_px e cm reais
    #   2) mm_per_px explícito
    #   3) px_per_mm explícito
    mm_per_px_effective = 0.0
    ruler_available = (
        p.ruler_y_top_px and p.ruler_y_bottom_px
        and p.ruler_y_bottom_px != p.ruler_y_top_px
        and p.ruler_bottom_cm != p.ruler_top_cm
    )

    if ruler_available:
        ruler_delta_mm = (float(p.ruler_bottom_cm) - float(p.ruler_top_cm)) * 10.0
        ruler_delta_px = float(p.ruler_y_bottom_px) - float(p.ruler_y_top_px)
        mm_per_px_effective = ruler_delta_mm / ruler_delta_px
    elif p.mm_per_px and p.mm_per_px > 0:
        mm_per_px_effective = float(p.mm_per_px)
    elif p.px_per_mm and p.px_per_mm > 0:
        mm_per_px_effective = 1.0 / float(p.px_per_mm)

    df["mm_per_px"] = mm_per_px_effective if mm_per_px_effective > 0 else np.nan

    # Coordenada absoluta na régua:
    # Se ruler_top_cm=15 e ruler_bottom_cm=17, a saída fica em mm de régua:
    # 150 mm, 160 mm, 170 mm...
    if ruler_available:
        ruler_top_mm = float(p.ruler_top_cm) * 10.0
        ruler_y_top_px = float(p.ruler_y_top_px)

        df["y_top_fixed_ruler_mm"] = ruler_top_mm + (df["y_top_fixed_px"] - ruler_y_top_px) * mm_per_px_effective
        df["y_center_ruler_mm"] = ruler_top_mm + (df["y_center_px"] - ruler_y_top_px) * mm_per_px_effective
        df["y_top_raw_ruler_mm"] = ruler_top_mm + (df["y_top_px"] - ruler_y_top_px) * mm_per_px_effective

        # Versão normalizada no intervalo 15→17 cm:
        # 0 mm na marca 15 cm; 20 mm na marca 17 cm.
        df["y_top_fixed_mm_from_ruler_top"] = (df["y_top_fixed_px"] - ruler_y_top_px) * mm_per_px_effective
        df["y_center_mm_from_ruler_top"] = (df["y_center_px"] - ruler_y_top_px) * mm_per_px_effective
    else:
        df["y_top_fixed_ruler_mm"] = np.nan
        df["y_center_ruler_mm"] = np.nan
        df["y_top_raw_ruler_mm"] = np.nan
        df["y_top_fixed_mm_from_ruler_top"] = np.nan
        df["y_center_mm_from_ruler_top"] = np.nan

    if mm_per_px_effective > 0:
        # Posição relativa em mm. Sinal positivo para cima.
        # Referência = primeiro frame válido.
        first_valid_idx = df.index[df["valid"].astype(bool) & df["y_top_fixed_px"].notna()]
        if len(first_valid_idx) > 0:
            ref_y_top = float(df.loc[first_valid_idx[0], "y_top_fixed_px"])
            ref_y_center = float(df.loc[first_valid_idx[0], "y_center_px"])
        else:
            ref_y_top = float(df["y_top_fixed_px"].dropna().iloc[0]) if df["y_top_fixed_px"].notna().any() else np.nan
            ref_y_center = float(df["y_center_px"].dropna().iloc[0]) if df["y_center_px"].notna().any() else np.nan

        df["z_top_fixed_mm_rel"] = (ref_y_top - df["y_top_fixed_px"]) * mm_per_px_effective
        df["z_center_mm_rel"] = (ref_y_center - df["y_center_px"]) * mm_per_px_effective
    else:
        df["z_top_fixed_mm_rel"] = np.nan
        df["z_center_mm_rel"] = np.nan


    # Suavização opcional em colunas extras, não substitui bruto.
    if p.median_smooth_window and p.median_smooth_window > 1:
        win = int(p.median_smooth_window)
        if win % 2 == 0:
            win += 1
        for col in ["x_center_px", "y_center_px", "radius_px", "y_top_px", "y_top_fixed_px", "z_top_fixed_mm_rel"]:
            df[col + "_median"] = df[col].rolling(win, center=True, min_periods=1).median()

    # atributos úteis
    df["roi_x_px"] = roi[0]
    df["roi_y_px"] = roi[1]
    df["roi_w_px"] = roi[2]
    df["roi_h_px"] = roi[3]
    df["video_fps"] = fps
    df["time_scale"] = p.time_scale

    df.to_csv(out_csv, index=False)

    if debug_candidates:
        pd.DataFrame(candidate_rows).to_csv(debug_candidates, index=False)

    if preview:
        make_preview(video_path, df, roi, preview, p)

    if plot:
        make_plot(df, plot)

    print("MAGLEV VIDEO TRACKER V6")
    print(f"Video: {video_path}")
    print(f"Resolution: {W}x{H}")
    print(f"FPS: {fps:.4f}")
    print(f"Frames: {n}")
    print(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print(f"Output CSV: {out_csv}")
    print(f"Valid frames: {int(df['valid'].sum())}/{len(df)}")
    if df["y_center_px"].notna().any():
        print(f"y_center_px range: {df['y_center_px'].min():.2f} .. {df['y_center_px'].max():.2f}")
        print(f"y_top_px range:    {df['y_top_px'].min():.2f} .. {df['y_top_px'].max():.2f}")
        print(f"confidence range:  {df['confidence'].min():.3f} .. {df['confidence'].max():.3f}")
        if "fixed_radius_px" in df.columns and df["fixed_radius_px"].notna().any():
            print(f"fixed_radius_px:   {df['fixed_radius_px'].dropna().iloc[0]:.2f}")
        if "mm_per_px" in df.columns and df["mm_per_px"].notna().any():
            print(f"mm_per_px:         {df['mm_per_px'].dropna().iloc[0]:.6f}")


def make_preview(video_path: str, df: pd.DataFrame, roi: Tuple[int, int, int, int], out_path: str, p: Params):
    meta = read_video_metadata(video_path)
    n = int(meta["frame_count"])
    idxs = np.linspace(0, max(0, n - 1), min(p.preview_frames, n)).astype(int)

    cap = cv2.VideoCapture(video_path)
    thumbs = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        row = df.iloc[int(idx)]
        vis = frame.copy()

        # ROI
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 180, 0), 2)

        if bool(row["valid"]) and not np.isnan(row["x_center_px"]):
            cx = int(round(row["x_center_px"]))
            cy = int(round(row["y_center_px"]))
            r = int(round(row["radius_px"]))
            yt = int(round(row["y_top_px"]))
            ytf = int(round(row["y_top_fixed_px"])) if "y_top_fixed_px" in row and not np.isnan(row["y_top_fixed_px"]) else yt
            rf = int(round(row["fixed_radius_px"])) if "fixed_radius_px" in row and not np.isnan(row["fixed_radius_px"]) else r

            # Verde: raio detectado bruto
            cv2.circle(vis, (cx, cy), max(2, r), (0, 255, 0), 2)
            # Magenta: raio fixo/metrológico
            if rf > 0:
                cv2.circle(vis, (cx, cy), max(2, rf), (255, 0, 255), 1)
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            # Amarelo: topo bruto por contorno
            cv2.line(vis, (cx-r, yt), (cx+r, yt), (0, 255, 255), 1)
            # Ciano: topo fixo recomendado
            cv2.line(vis, (cx-rf, ytf), (cx+rf, ytf), (255, 255, 0), 2)
            text = f"f={idx} ytop_fix={row['y_top_fixed_px']:.1f} conf={row['confidence']:.2f}"
        else:
            text = f"f={idx} INVALID conf={row['confidence']:.2f}"

        cv2.putText(vis, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # resize thumb
        thumb_w = 360
        scale = thumb_w / vis.shape[1]
        thumb = cv2.resize(vis, (thumb_w, int(vis.shape[0] * scale)))
        thumbs.append(thumb)

    cap.release()

    if not thumbs:
        return

    # grid
    cols = 3
    rows = int(math.ceil(len(thumbs) / cols))
    th, tw = thumbs[0].shape[:2]
    canvas = np.full((rows * th, cols * tw, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        canvas[r*th:r*th+thumb.shape[0], c*tw:c*tw+thumb.shape[1]] = thumb

    cv2.imwrite(out_path, canvas)


def make_plot(df: pd.DataFrame, out_path: str):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 7))
    ax = plt.gca()

    valid = df["valid"].astype(bool)
    ax.plot(df["time_real_s"], df["y_center_px"], label="y_center_px bruto", linewidth=1.2)
    ax.plot(df["time_real_s"], df["y_top_px"], label="y_top_px bruto", linewidth=0.9, alpha=0.55)
    if "y_top_fixed_px" in df.columns:
        ax.plot(df["time_real_s"], df["y_top_fixed_px"], label="y_top_fixed_px recomendado", linewidth=2.0)
    ax.scatter(df.loc[~valid, "time_real_s"], df.loc[~valid, "y_center_px"], label="invalid", s=18)

    if "y_top_px_median" in df.columns:
        ax.plot(df["time_real_s"], df["y_top_px_median"], label="y_top_px median", linewidth=2)

    ax.set_xlabel("tempo real estimado (s)")
    ax.set_ylabel("posição vertical (px)")
    ax.set_title("Maglev video tracking V6")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def load_config(path: str) -> Params:
    with open(path, "r", encoding="utf-8") as f:
        d = json_load_strip_comments(f.read())
    p = Params()
    for k, v in d.items():
        if hasattr(p, k):
            setattr(p, k, v)
    if isinstance(p.roi, list):
        p.roi = tuple(p.roi)  # type: ignore
    return p


def json_load_strip_comments(s: str) -> dict:
    # JSON simples com suporte tosco a linhas começando com //
    import json
    lines = []
    for line in s.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))


def save_default_config(path: str):
    import json
    p = Params()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(p), f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Caminho do vídeo de entrada")
    ap.add_argument("--out", default=None, help="CSV de saída. Se omitido, salva em <nome_do_video>/tracking.csv")
    ap.add_argument("--output-dir", default=None, help="Pasta de saída. Se omitida, usa <nome_do_video>/")
    ap.add_argument("--preview", default=None, help="Imagem JPG/PNG com frames anotados. Se omitido, salva em <output-dir>/preview.jpg")
    ap.add_argument("--plot", default=None, help="Gráfico PNG da trajetória")
    ap.add_argument("--debug-candidates", default=None, help="CSV opcional com candidatos por frame")
    ap.add_argument("--config", default=None, help="JSON de parâmetros")
    ap.add_argument("--write-default-config", default=None, help="Escreve JSON default e sai")
    ap.add_argument("--roi", default=None, help="ROI manual x,y,w,h. Se usado, desliga auto_roi.")
    ap.add_argument("--time-scale", type=float, default=None, help="Fator tempo_real = tempo_video * time_scale")
    ap.add_argument("--median-smooth-window", type=int, default=None, help="Janela mediana opcional para colunas extras")
    ap.add_argument("--fixed-radius-px", type=float, default=None, help="Raio fixo da bolinha em pixels")
    ap.add_argument("--ball-diameter-mm", type=float, default=None, help="Diâmetro real da bolinha em mm")
    ap.add_argument("--px-per-mm", type=float, default=None, help="Escala do vídeo: pixels por mm")
    ap.add_argument("--mm-per-px", type=float, default=None, help="Escala do vídeo: mm por pixel")
    ap.add_argument("--ruler-y-top-px", type=float, default=None, help="Pixel Y da marca superior da régua, ex: marca 15 cm")
    ap.add_argument("--ruler-y-bottom-px", type=float, default=None, help="Pixel Y da marca inferior da régua, ex: marca 17 cm")
    ap.add_argument("--ruler-top-cm", type=float, default=None, help="Valor em cm da marca superior, padrão 15")
    ap.add_argument("--ruler-bottom-cm", type=float, default=None, help="Valor em cm da marca inferior, padrão 17")
    ap.add_argument("--no-auto-fixed-radius", action="store_true", help="Desliga cálculo automático do raio fixo pela mediana")
    args = ap.parse_args()

    if args.write_default_config:
        save_default_config(args.write_default_config)
        print(f"Config default salva em: {args.write_default_config}")
        return

    p = load_config(args.config) if args.config else Params()

    if args.roi:
        p.roi = parse_tuple4(args.roi)
        p.auto_roi = False
    if args.time_scale is not None:
        p.time_scale = args.time_scale
    if args.median_smooth_window is not None:
        p.median_smooth_window = args.median_smooth_window
    if args.fixed_radius_px is not None:
        p.fixed_radius_px = args.fixed_radius_px
    if args.ball_diameter_mm is not None:
        p.ball_diameter_mm = args.ball_diameter_mm
    if args.px_per_mm is not None:
        p.px_per_mm = args.px_per_mm
    if args.mm_per_px is not None:
        p.mm_per_px = args.mm_per_px
    if args.ruler_y_top_px is not None:
        p.ruler_y_top_px = args.ruler_y_top_px
    if args.ruler_y_bottom_px is not None:
        p.ruler_y_bottom_px = args.ruler_y_bottom_px
    if args.ruler_top_cm is not None:
        p.ruler_top_cm = args.ruler_top_cm
    if args.ruler_bottom_cm is not None:
        p.ruler_bottom_cm = args.ruler_bottom_cm
    if args.no_auto_fixed_radius:
        p.auto_fixed_radius = False

    # ------------------------------------------------------------
    # Saídas automáticas por vídeo
    # ------------------------------------------------------------
    # Exemplo:
    #   --video ident_slow_mo.mp4
    # cria:
    #   ident_slow_mo/
    #       tracking.csv
    #       preview.jpg
    #       plot.png
    #       candidates.csv
    #
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = args.output_dir if args.output_dir else video_stem
    os.makedirs(output_dir, exist_ok=True)

    out_csv = args.out if args.out else os.path.join(output_dir, "tracking.csv")
    preview = args.preview if args.preview else os.path.join(output_dir, "preview.jpg")
    plot = args.plot if args.plot else os.path.join(output_dir, "plot.png")
    debug_candidates = args.debug_candidates if args.debug_candidates else os.path.join(output_dir, "candidates.csv")

    process_video(
        video_path=args.video,
        out_csv=out_csv,
        p=p,
        preview=preview,
        plot=plot,
        debug_candidates=debug_candidates,
    )

    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
