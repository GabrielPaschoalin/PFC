#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Escurece seletivamente a bolinha do vídeo do Maglev para facilitar o tracking
por blob escuro nas versões v31/v32 do tracker.

Ideia:
  1) detectar a linha verde/ciano da régua para saber onde a bolinha NÃO pode estar;
  2) detectar a bolinha por Hough + máscara escura, com consistência temporal;
  3) escurecer apenas a região estimada da bolinha, preservando régua, LED e fundo.

Dependências:
  pip install opencv-python numpy

Exemplo:
  python escurecer_bolinha_maglev.py --input 190.mp4 --output 190_bolinha_preta.mp4 --preview

Modo mais agressivo:
  python escurecer_bolinha_maglev.py --input 190.mp4 --output 190_bolinha_preta_agressivo.mp4 --aggressive --preview
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Params:
    # Régua / exclusões
    ruler_search_x0_ratio: float = 0.50
    ruler_green_h_min: int = 35
    ruler_green_h_max: int = 100
    ruler_green_s_min: int = 45
    ruler_green_v_min: int = 60
    ruler_left_margin_px: int = 28
    led_exclusion_dilate_px: int = 15

    # Hough para estimar a região circular da bolinha
    hough_dp: float = 1.2
    hough_min_dist: int = 85
    hough_param1: int = 80
    hough_param2: int = 18
    min_radius_px: int = 55
    max_radius_px: int = 145
    max_jump_px: float = 90.0

    # Máscara escura dentro da região da bolinha
    dark_abs_threshold: int = 135
    dark_min_threshold: int = 75
    dark_percentile: float = 10.0
    mask_open_px: int = 3
    mask_close_px: int = 25
    mask_dilate_px: int = 5
    min_ball_area_px: int = 4500
    max_ball_area_ratio: float = 0.38
    min_ball_width_px: int = 55
    min_ball_height_px: int = 55
    max_aspect_ratio: float = 2.5

    # Escurecimento
    strength: float = 0.95          # 0.0 = não muda; 1.0 = aplica totalmente o preto
    black_level: int = 0            # 0 = preto puro; 20/30 = preto menos artificial
    feather_px: int = 5             # suavização de borda da máscara
    force_circle: bool = False      # True: pinta o círculo inteiro; False: pinta só o contorno estimado

    # Debug/preview
    preview_frames: int = 18
    preview_width_px: int = 360
    detect_every_frames: int = 5       # roda Hough a cada N frames; nos intermediários reutiliza a estimativa


def odd_at_least(value: int, minimum: int = 3) -> int:
    v = max(int(value), int(minimum))
    return v if v % 2 == 1 else v + 1


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


def green_guide_mask(frame: np.ndarray, p: Params) -> np.ndarray:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    if p.ruler_green_h_min <= p.ruler_green_h_max:
        hue_ok = (hue >= p.ruler_green_h_min) & (hue <= p.ruler_green_h_max)
    else:
        hue_ok = (hue >= p.ruler_green_h_min) | (hue <= p.ruler_green_h_max)

    mask = (hue_ok & (sat >= p.ruler_green_s_min) & (val >= p.ruler_green_v_min)).astype(np.uint8) * 255
    mask[:, : int(w * p.ruler_search_x0_ratio)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)), iterations=1)
    return mask


def detect_ruler_x_limit(frames: List[np.ndarray], p: Params) -> Optional[int]:
    xs = []
    for frame in frames:
        mask = green_guide_mask(frame, p)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            area = float(cv2.contourArea(c))
            x, y, w, h = cv2.boundingRect(c)
            if area < 80 or h < 25:
                continue
            score = area + 30.0 * h + 80.0 * (h / max(1.0, w)) + 0.02 * x
            if best is None or score > best[0]:
                best = (score, x, y, w, h)
        if best is not None:
            _, x, _y, w, _h = best
            xs.append(float(x + w / 2.0))

    if not xs:
        return None
    guide_x = float(np.median(xs))
    return int(round(guide_x - p.ruler_left_margin_px))


def build_allowed_mask(frame: np.ndarray, x_limit: Optional[int], p: Params) -> np.ndarray:
    h, w = frame.shape[:2]
    allowed = np.ones((h, w), dtype=np.uint8) * 255

    if x_limit is not None:
        x_limit = max(1, min(w, int(x_limit)))
        allowed[:, x_limit:] = 0

    red = red_led_mask(frame)
    d = int(p.led_exclusion_dilate_px)
    if d > 0:
        red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_at_least(d), odd_at_least(d))), iterations=1)
    allowed[red > 0] = 0
    return allowed


def dark_threshold(gray_blur: np.ndarray, allowed: np.ndarray, p: Params) -> float:
    vals = gray_blur[allowed > 0]
    if vals.size == 0:
        return float(p.dark_abs_threshold)
    q = float(np.percentile(vals, p.dark_percentile))
    return float(min(p.dark_abs_threshold, max(q, p.dark_min_threshold)))


def make_dark_mask(frame: np.ndarray, allowed: np.ndarray, p: Params) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = dark_threshold(blur, allowed, p)
    mask = (blur <= thr).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, allowed)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_at_least(p.mask_open_px), odd_at_least(p.mask_open_px))), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_at_least(p.mask_close_px), odd_at_least(p.mask_close_px))), iterations=1)
    return mask


def circle_mask(shape: Tuple[int, int], circle: Tuple[float, float, float], scale: float = 1.0) -> np.ndarray:
    h, w = shape
    cx, cy, r = circle
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r * scale)), 255, -1)
    return mask


def score_circle(frame: np.ndarray, circle: Tuple[float, float, float], allowed: np.ndarray, prev: Optional[Tuple[float, float, float]], p: Params) -> float:
    h, w = frame.shape[:2]
    cx, cy, r = circle
    if cx - r < -20 or cy - r < -40 or cx + r > w + 20 or cy + r > h + 40:
        return -1e9
    if allowed[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))] == 0:
        return -1e9

    cm = circle_mask((h, w), circle, 0.92)
    cm = cv2.bitwise_and(cm, allowed)
    if cv2.countNonZero(cm) < 100:
        return -1e9

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_inside = float(cv2.mean(gray, mask=cm)[0])
    dark_score = np.clip((170.0 - mean_inside) / 120.0, 0.0, 1.0)

    # Mede quanto da região circular já é escura. Ajuda a evitar círculos falsos na régua/fundo claro.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = dark_threshold(blur, allowed, p)
    dm = ((blur <= thr).astype(np.uint8) * 255)
    overlap = cv2.countNonZero(cv2.bitwise_and(dm, cm)) / max(1, cv2.countNonZero(cm))

    temporal = 0.0
    if prev is not None:
        px, py, pr = prev
        dist = math.hypot(cx - px, cy - py)
        if dist > p.max_jump_px:
            temporal = -2.0 - 0.02 * dist
        else:
            temporal = 1.4 * (1.0 - dist / max(1.0, p.max_jump_px))
        radius_score = 0.4 * (1.0 - min(1.0, abs(r - pr) / max(1.0, pr)))
    else:
        # No início, tende a preferir a bolinha mais central/baixa, não os círculos falsos do topo.
        radius_score = 0.2 * np.clip((r - p.min_radius_px) / max(1, p.max_radius_px - p.min_radius_px), 0.0, 1.0)
        temporal = 0.25 * np.clip(cy / max(1, h), 0.0, 1.0)

    return 1.6 * dark_score + 1.7 * overlap + temporal + radius_score


def detect_ball_circle(frame: np.ndarray, allowed: np.ndarray, prev: Optional[Tuple[float, float, float]], p: Params) -> Optional[Tuple[float, float, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aceleração: roda o Hough em uma imagem reduzida e somente na área permitida
    # à esquerda da régua. Isso mantém o script viável em vídeos longos.
    ys, xs = np.where(allowed > 0)
    if xs.size == 0 or ys.size == 0:
        return prev
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    crop = gray[y0:y1, x0:x1].copy()
    crop_allowed = allowed[y0:y1, x0:x1]
    crop[crop_allowed == 0] = 255

    det_scale = 0.50
    small = cv2.resize(crop, None, fx=det_scale, fy=det_scale, interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)

    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=float(p.hough_dp),
        minDist=max(20, int(p.hough_min_dist * det_scale)),
        param1=int(p.hough_param1),
        param2=int(p.hough_param2),
        minRadius=max(10, int(p.min_radius_px * det_scale)),
        maxRadius=max(12, int(p.max_radius_px * det_scale)),
    )
    if circles is None:
        return prev

    best = None
    inv_scale = 1.0 / det_scale
    for c in circles[0, :]:
        cx = x0 + float(c[0]) * inv_scale
        cy = y0 + float(c[1]) * inv_scale
        r = float(c[2]) * inv_scale
        sc = score_circle(frame, (cx, cy, r), allowed, prev, p)
        if best is None or sc > best[0]:
            best = (sc, cx, cy, r)

    if best is None or best[0] < -1e8:
        return prev

    _, cx, cy, r = best

    # Suavização simples: reduz tremulação no vídeo tratado.
    if prev is not None:
        px, py, pr = prev
        alpha = 0.35
        cx = alpha * cx + (1.0 - alpha) * px
        cy = alpha * cy + (1.0 - alpha) * py
        r = alpha * r + (1.0 - alpha) * pr

    return (cx, cy, r)


def largest_contour_mask(mask: np.ndarray, p: Params) -> Optional[np.ndarray]:
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = p.max_ball_area_ratio * float(w * h)
    best = None
    for c in contours:
        area = float(cv2.contourArea(c))
        x, y, bw, bh = cv2.boundingRect(c)
        if area < p.min_ball_area_px or area > max_area:
            continue
        if bw < p.min_ball_width_px or bh < p.min_ball_height_px:
            continue
        if bw / max(1.0, float(bh)) > p.max_aspect_ratio:
            continue
        fill = area / max(1.0, float(bw * bh))
        score = area + 1000.0 * fill - 30.0 * abs((bw / max(1, bh)) - 1.0)
        if best is None or score > best[0]:
            best = (score, c)
    if best is None:
        return None
    out = np.zeros_like(mask)
    cv2.drawContours(out, [best[1]], -1, 255, -1)
    return out


def build_ball_adjustment_mask(frame: np.ndarray, allowed: np.ndarray, circle: Optional[Tuple[float, float, float]], p: Params) -> Tuple[np.ndarray, str]:
    h, w = frame.shape[:2]
    if circle is None:
        # Fallback puramente por máscara escura.
        dark = make_dark_mask(frame, allowed, p)
        chosen = largest_contour_mask(dark, p)
        return (chosen if chosen is not None else np.zeros((h, w), dtype=np.uint8), "fallback_contour")

    cm = circle_mask((h, w), circle, 1.02)
    cm = cv2.bitwise_and(cm, allowed)
    if p.force_circle:
        return cm, "force_circle"

    dark = make_dark_mask(frame, allowed, p)
    local = cv2.bitwise_and(dark, cm)
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_at_least(p.mask_close_px), odd_at_least(p.mask_close_px))), iterations=1)

    chosen = largest_contour_mask(local, p)
    if chosen is None:
        # Se a máscara escura falhar, usa o círculo, mas um pouco menor.
        return circle_mask((h, w), circle, 0.92), "circle_fallback"

    if p.mask_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_at_least(p.mask_dilate_px), odd_at_least(p.mask_dilate_px)))
        chosen = cv2.dilate(chosen, k, iterations=1)
        chosen = cv2.bitwise_and(chosen, cm)
    return chosen, "contour_in_circle"


def apply_blackening(frame: np.ndarray, mask: np.ndarray, p: Params) -> np.ndarray:
    if mask is None or cv2.countNonZero(mask) == 0:
        return frame

    alpha = mask.astype(np.float32) / 255.0
    if p.feather_px > 0:
        k = odd_at_least(p.feather_px)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

    alpha = alpha[:, :, None] * float(np.clip(p.strength, 0.0, 1.0))
    target = np.full_like(frame, int(np.clip(p.black_level, 0, 255)))
    out = frame.astype(np.float32) * (1.0 - alpha) + target.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    panel_h = 32
    panel = np.full((panel_h, img.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(panel, text[:120], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([img, panel])


def make_preview(input_path: Path, output_path: Path, preview_rows: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, str]], p: Params) -> None:
    thumbs = []
    for frame_idx, original, adjusted, mask, mode in preview_rows:
        vis = adjusted.copy()
        overlay = vis.copy()
        overlay[mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)
        side = np.hstack([original, vis])
        scale = float(p.preview_width_px * 2) / side.shape[1]
        side = cv2.resize(side, (int(side.shape[1] * scale), int(side.shape[0] * scale)))
        side = add_label(side, f"f={frame_idx} | esq=original | dir=processado+mascara | modo={mode}")
        thumbs.append(side)

    if not thumbs:
        return
    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    th, tw = thumbs[0].shape[:2]
    canvas = np.full((rows * th, cols * tw, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        canvas[r * th : r * th + thumb.shape[0], c * tw : c * tw + thumb.shape[1]] = thumb
    cv2.imwrite(str(output_path), canvas)


def process_video(input_path: Path, output_path: Path, preview_path: Optional[Path], p: Params) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir o vídeo: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Amostra alguns frames iniciais para localizar a linha verde/ciano da régua.
    sampled = []
    sample_n = min(25, max(1, frame_count))
    for idx in np.linspace(0, max(0, frame_count - 1), sample_n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            sampled.append(frame)
    x_limit = detect_ruler_x_limit(sampled, p)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Não consegui criar o vídeo de saída: {output_path}")

    preview_idxs = set(np.linspace(0, max(0, frame_count - 1), min(p.preview_frames, max(1, frame_count))).astype(int).tolist())
    preview_rows = []

    prev_circle: Optional[Tuple[float, float, float]] = None
    processed = 0
    with_mask = 0
    mode_counts = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        allowed = build_allowed_mask(frame, x_limit, p)
        # Hough é a etapa mais cara. Como o movimento entre frames é suave,
        # detecta periodicamente e reutiliza a última estimativa nos frames intermediários.
        if prev_circle is None or ((processed - 1) % max(1, int(p.detect_every_frames)) == 0):
            circle = detect_ball_circle(frame, allowed, prev_circle, p)
            if circle is not None:
                prev_circle = circle
        else:
            circle = prev_circle
        mask, mode = build_ball_adjustment_mask(frame, allowed, circle, p)
        adjusted = apply_blackening(frame, mask, p)
        writer.write(adjusted)

        processed += 1
        if cv2.countNonZero(mask) > 0:
            with_mask += 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        frame_idx = processed - 1
        if preview_path is not None and frame_idx in preview_idxs:
            preview_rows.append((frame_idx, frame.copy(), adjusted.copy(), mask.copy(), mode))

    cap.release()
    writer.release()

    if preview_path is not None:
        make_preview(input_path, preview_path, preview_rows, p)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "preview": str(preview_path) if preview_path else "",
        "fps": fps,
        "resolution": f"{width}x{height}",
        "frames": processed,
        "frames_com_mascara": with_mask,
        "ruler_x_limit": x_limit,
        "modos": mode_counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Escurece seletivamente a bolinha do vídeo do Maglev.")
    ap.add_argument("--input", required=True, type=Path, help="Vídeo de entrada, ex.: 190.mp4")
    ap.add_argument("--output", required=True, type=Path, help="Vídeo de saída, ex.: 190_bolinha_preta.mp4")
    ap.add_argument("--preview", action="store_true", help="Gera uma imagem JPG comparando original/processado")
    ap.add_argument("--preview-path", type=Path, default=None, help="Caminho da imagem de preview")
    ap.add_argument("--aggressive", action="store_true", help="Escurecimento mais forte e máscara mais expansiva")
    ap.add_argument("--force-circle", action="store_true", help="Pinta o círculo inteiro detectado, mais agressivo")
    ap.add_argument("--strength", type=float, default=None, help="Força do escurecimento, 0..1")
    ap.add_argument("--black-level", type=int, default=None, help="Nível de preto final. 0=preto puro")
    ap.add_argument("--dark-threshold", type=int, default=None, help="Threshold absoluto da máscara escura")
    ap.add_argument("--hough-param2", type=int, default=None, help="Sensibilidade do Hough. Menor = mais círculos")
    ap.add_argument("--detect-every", type=int, default=None, help="Roda Hough a cada N frames. Default: 5")
    args = ap.parse_args()

    p = Params()
    if args.aggressive:
        p.strength = 1.0
        p.black_level = 0
        p.mask_close_px = 31
        p.mask_dilate_px = 9
        p.dark_abs_threshold = 150
        p.hough_param2 = 15
    if args.force_circle:
        p.force_circle = True
    if args.strength is not None:
        p.strength = float(args.strength)
    if args.black_level is not None:
        p.black_level = int(args.black_level)
    if args.dark_threshold is not None:
        p.dark_abs_threshold = int(args.dark_threshold)
    if args.hough_param2 is not None:
        p.hough_param2 = int(args.hough_param2)
    if args.detect_every is not None:
        p.detect_every_frames = max(1, int(args.detect_every))

    preview_path = None
    if args.preview:
        preview_path = args.preview_path or args.output.with_name(args.output.stem + "_preview.jpg")

    resumo = process_video(args.input, args.output, preview_path, p)
    print("Resumo do processamento:")
    for k, v in resumo.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
