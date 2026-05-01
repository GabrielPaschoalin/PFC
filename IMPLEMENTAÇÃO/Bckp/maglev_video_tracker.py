#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAGLEV VIDEO TRACKER V10 - V9 + TRACKING TUNING + CACHE

Objetivo:
    Ler um vídeo do Maglev e gerar CSVs com a posição da bolinha ao longo do tempo,
    usando o rastreamento da V6 e adicionando pós-processamento de sincronismo,
    tempo em milissegundos e calibração automática da régua inclinada.

    V10: mantém a base da V9 e adiciona ajustes de robustez para a pedra irregular,
    restrição automática da ROI pela régua, cache de background e leitura assíncrona
    de frames para reduzir o tempo de processamento.

Saídas:
    tracking_full.csv       -> vídeo completo
    tracking_tratado.csv    -> trecho sincronizado a partir da borda de subida do LED
    preview.jpg             -> preview do rastreamento da bolinha
    plot.png                -> gráfico do CSV tratado
    candidates.csv          -> candidatos por frame
    sync_preview.jpg        -> conferência do LED de sincronismo
    ruler_preview.jpg       -> conferência da régua detectada automaticamente

Dependências:
    pip install opencv-python numpy pandas matplotlib

Opcional:
    ffprobe disponível no PATH para leitura mais completa dos metadados do vídeo.

Exemplo:
    python maglev_video_tracker_v10.py --video ident_slow_mo.mp4
"""


from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import threading
import queue
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
    # Campo legado mantido apenas para compatibilidade de configuração.
    # A V8 usa metadados do vídeo e, quando necessário, a largura conhecida do pulso do LED.
    time_scale: float = 1.0

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

    # Campos legados para fallback manual, caso a detecção automática da régua falhe.
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
    motion_sample_count: int = 160
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
    min_circularity: float = 0.20
    min_fill_ratio: float = 0.18
    max_fill_ratio: float = 1.35
    max_aspect_deviation: float = 0.80
    # aspect_deviation = abs(w-h)/max(w,h)

    # -------------------------
    # Fusão / pontuação
    # -------------------------
    # Score candidato = combinação ponderada.
    w_brightness: float = 0.12
    w_motion: float = 0.38
    w_circularity: float = 0.14
    w_size: float = 0.10
    w_roi_center_prior: float = 0.04
    w_template: float = 0.22

    # Template opcional, atualizado a partir do melhor candidato do primeiro trecho.
    # Use 0 se quiser desligar.
    template_update_alpha: float = 0.08
    template_min_confidence_to_update: float = 0.52

    # -------------------------
    # Associação temporal
    # -------------------------
    # Programação dinâmica: favorece continuidade, mas permite saltos reais.
    max_jump_px_soft: float = 30.0
    jump_penalty_weight: float = 0.035
    missing_penalty: float = 0.55

    # Se o objeto for perdido, ainda assim escolhe o melhor candidato local/global.
    max_candidates_per_frame: int = 8

    # -------------------------
    # Validade / confiança
    # -------------------------
    valid_confidence_threshold: float = 0.48

    # -------------------------
    # Pós-processamento opcional
    # -------------------------
    # Não altera y_center_px/y_top_px bruto. Cria colunas suavizadas se > 1.
    median_smooth_window: int = 7

    # -------------------------
    # Sincronismo por LED vermelho
    # -------------------------
    sync_enabled: bool = True
    sync_crop_to_rising_edge: bool = True
    sync_known_pulse_width_ms: float = 200.0
    sync_red_min_area_px: int = 18
    sync_score_threshold: float = 0.50
    sync_temporal_smooth_frames: int = 2

    # -------------------------
    # Calibração automática da régua
    # -------------------------
    auto_ruler: bool = True
    ruler_search_right_ratio: float = 0.38
    ruler_tick_min_spacing_px: float = 4.0
    ruler_tick_max_spacing_px: float = 28.0
    ruler_preview_ticks: int = 80

    # -------------------------
    # Preview
    # -------------------------
    preview_frames: int = 18
    preview_width_px: int = 1800

    # -------------------------
    # Aceleração por GPU
    # -------------------------
    # A V9 usa OpenCV CUDA quando disponível para pré-processamento de frames.
    # Se a instalação do OpenCV não tiver CUDA, volta automaticamente para CPU.
    use_gpu: bool = True
    gpu_device_id: int = 0

    # -------------------------
    # Otimizações V10
    # -------------------------
    background_cache_enabled: bool = True
    background_percentile: float = 30.0
    frame_reader_thread: bool = True
    frame_queue_size: int = 32
    auto_exclude_ruler_from_roi: bool = True
    roi_ruler_margin_px: int = 10
    physical_outlier_enabled: bool = True
    max_physical_delta_mm_per_frame: float = 2.0


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



class FrameGpuAccelerator:
    """
    Acelerador opcional de pré-processamento.

    Mantém as mesmas entradas e saídas usadas pela lógica da V8. As etapas de
    decisão continuam iguais; quando CUDA não está disponível, o fallback CPU
    executa exatamente o caminho original.
    """

    def __init__(self, p: Params):
        self.enabled = False
        self.reason = "disabled"
        self.device_id = int(getattr(p, "gpu_device_id", 0))
        self._gaussian_filters: Dict[Tuple[int, int, int], object] = {}
        self._morph_filters: Dict[Tuple[int, int, Tuple[int, int]], object] = {}
        self._hough_detectors: Dict[Tuple[float, int, int, int, int, int], object] = {}
        self.gpu_fallback_count = 0

        if not getattr(p, "use_gpu", True):
            self.reason = "disabled_by_user"
            return

        try:
            if not hasattr(cv2, "cuda"):
                self.reason = "opencv_without_cuda_module"
                return
            count = int(cv2.cuda.getCudaEnabledDeviceCount())
            if count <= 0:
                self.reason = "no_cuda_device_or_opencv_without_cuda"
                return
            if self.device_id < 0 or self.device_id >= count:
                self.device_id = 0
            cv2.cuda.setDevice(self.device_id)
            # Checagem leve das funções usadas pela V9.
            if not hasattr(cv2.cuda, "cvtColor") or not hasattr(cv2.cuda, "absdiff"):
                self.reason = "cuda_api_incomplete"
                return
            _ = cv2.cuda_GpuMat()
            self.enabled = True
            self.reason = f"cuda_device_{self.device_id}"
        except Exception as exc:
            self.enabled = False
            self.reason = f"cuda_unavailable:{type(exc).__name__}"

    def _filter(self, ksize: Tuple[int, int], sigma: float):
        key = (int(ksize[0]), int(ksize[1]), int(round(float(sigma) * 1000)))
        filt = self._gaussian_filters.get(key)
        if filt is None:
            filt = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, ksize, sigma)
            self._gaussian_filters[key] = filt
        return filt

    def preprocess_ball(self, frame_roi: np.ndarray, bg_roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna gray, gray_blur e diff_blur em CPU, preservando a lógica de
        segmentação da V8.
        """
        if not self.enabled:
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, bg_roi)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
            return gray, gray_blur, diff_blur

        try:
            g_frame = cv2.cuda_GpuMat()
            g_frame.upload(frame_roi)
            g_gray = cv2.cuda.cvtColor(g_frame, cv2.COLOR_BGR2GRAY)

            g_bg = cv2.cuda_GpuMat()
            g_bg.upload(bg_roi)
            g_diff = cv2.cuda.absdiff(g_gray, g_bg)

            filt = self._filter((5, 5), 0.0)
            g_gray_blur = filt.apply(g_gray)
            g_diff_blur = filt.apply(g_diff)

            gray = g_gray.download()
            gray_blur = g_gray_blur.download()
            diff_blur = g_diff_blur.download()
            return gray, gray_blur, diff_blur
        except Exception:
            self.gpu_fallback_count += 1
            self.enabled = False
            self.reason = "runtime_cuda_fallback_cpu"
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, bg_roi)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
            return gray, gray_blur, diff_blur

    def hsv(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        try:
            g_frame = cv2.cuda_GpuMat()
            g_frame.upload(frame)
            g_hsv = cv2.cuda.cvtColor(g_frame, cv2.COLOR_BGR2HSV)
            return g_hsv.download()
        except Exception:
            self.gpu_fallback_count += 1
            self.enabled = False
            self.reason = "runtime_cuda_fallback_cpu"
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def morphology(self, mask: np.ndarray, op: int, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
        if not self.enabled or not hasattr(cv2.cuda, "createMorphologyFilter"):
            return cv2.morphologyEx(mask, op, kernel, iterations=iterations)
        try:
            g_mask = cv2.cuda_GpuMat()
            g_mask.upload(mask)
            kshape = tuple(int(v) for v in kernel.shape[:2])
            key = (int(op), int(cv2.CV_8UC1), kshape)
            filt = self._morph_filters.get(key)
            if filt is None:
                filt = cv2.cuda.createMorphologyFilter(op, cv2.CV_8UC1, kernel)
                self._morph_filters[key] = filt
            out = g_mask
            for _ in range(max(1, int(iterations))):
                out = filt.apply(out)
            return out.download()
        except Exception:
            self.gpu_fallback_count += 1
            self.enabled = False
            self.reason = "runtime_cuda_fallback_cpu"
            return cv2.morphologyEx(mask, op, kernel, iterations=iterations)

    def hough_circles(self, gray_blur: np.ndarray, p: Params) -> Optional[np.ndarray]:
        if not self.enabled or not hasattr(cv2.cuda, "createHoughCirclesDetector"):
            return None
        try:
            g_gray = cv2.cuda_GpuMat()
            g_gray.upload(gray_blur)
            key = (1.2, max(20, int(p.min_ball_radius_px)), 80, 18, int(p.min_ball_radius_px), int(p.max_ball_radius_px))
            detector = self._hough_detectors.get(key)
            if detector is None:
                detector = cv2.cuda.createHoughCirclesDetector(*key)
                self._hough_detectors[key] = detector
            circles_gpu = detector.detect(g_gray)
            if circles_gpu is None:
                return None
            if hasattr(circles_gpu, "download"):
                circles = circles_gpu.download()
            else:
                circles = np.asarray(circles_gpu)
            if circles is None or np.asarray(circles).size == 0:
                return np.empty((0, 3), dtype=float)
            circles = np.asarray(circles, dtype=float).reshape(-1, 3)
            return circles
        except Exception:
            # Mantém CUDA ligado para as demais etapas; apenas Hough volta para CPU.
            self.gpu_fallback_count += 1
            return None



def frame_reader_worker(video_path: str, out_queue: "queue.Queue", maxsize: int = 32):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out_queue.put((idx, frame))
            idx += 1
    finally:
        cap.release()
        out_queue.put(None)


def iter_video_frames(video_path: str, p: Params):
    if not getattr(p, "frame_reader_thread", True):
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
        return

    qsize = max(1, int(getattr(p, "frame_queue_size", 32)))
    q: "queue.Queue" = queue.Queue(maxsize=qsize)
    reader = threading.Thread(target=frame_reader_worker, args=(video_path, q, qsize), daemon=True)
    reader.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    reader.join(timeout=1.0)


def restrict_roi_by_ruler(roi: Tuple[int, int, int, int], ruler: dict, W: int, H: int, p: Params) -> Tuple[int, int, int, int]:
    if not getattr(p, "auto_exclude_ruler_from_roi", True):
        return roi
    if not ruler.get("auto_ruler_ok", False):
        return roi
    ruler_x = int(ruler.get("ruler_roi_x_px", W))
    x0, y0, w0, h0 = roi
    new_w = min(w0, ruler_x - x0 - int(getattr(p, "roi_ruler_margin_px", 10)))
    if new_w > 50:
        return clamp_roi((x0, y0, int(new_w), h0), W, H, p.frame_border_ignore_px)
    return roi


def flag_physical_outliers(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy()
    df["physical_outlier"] = False
    df["valid_signal"] = df["valid"].astype(bool) if "valid" in df.columns else True
    if not getattr(p, "physical_outlier_enabled", True):
        return df

    candidate_cols = [
        "z_top_fixed_mm_rel_corr",
        "z_top_fixed_mm_rel",
        "y_top_fixed_px",
    ]
    col = next((c for c in candidate_cols if c in df.columns and df[c].notna().any()), None)
    if col is None:
        return df

    vals = df[col].to_numpy(dtype=float)
    valid = df["valid"].astype(bool).to_numpy() if "valid" in df.columns else np.ones(len(df), dtype=bool)
    out = np.zeros(len(df), dtype=bool)
    max_delta = float(getattr(p, "max_physical_delta_mm_per_frame", 2.0))
    last_val = np.nan
    for i, z in enumerate(vals):
        if not valid[i] or not np.isfinite(z):
            continue
        if np.isfinite(last_val) and abs(z - last_val) > max_delta:
            out[i] = True
        else:
            last_val = z
    df["physical_outlier"] = out
    df["valid_signal"] = valid & (~out)
    return df


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
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
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

    keys_priority = [
        "com.android.capture.fps",
        "com.android.capture.framerate",
        "com.samsung.android.capture.fps",
        "captureframerate",
        "capture_framerate",
        "capture fps",
        "originalframerate",
        "original_frame_rate",
        "slow_motion_capture_fps",
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
        for key in keys_priority:
            if key in lowered:
                fps = _parse_ratio(lowered[key])
                if fps > 1:
                    return fps, f"ffprobe_tag:{key}"

    return 0.0, "none"


def read_video_metadata(video_path: str) -> Dict[str, object]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir o vídeo: {video_path}")

    fps_cv = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        confidence = "LOW"

    duration_s = float(n / fps_playback if fps_playback else 0.0)
    time_scale_detected = float(fps_playback / fps_capture) if fps_capture else np.nan

    return {
        "fps": float(fps_playback),
        "fps_playback": float(fps_playback),
        "fps_capture": float(fps_capture),
        "frame_count": int(n),
        "width": int(W),
        "height": int(H),
        "duration_s": duration_s,
        "time_scale_detected": time_scale_detected,
        "metadata_time_source": source,
        "metadata_time_confidence": confidence,
    }

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


def estimate_background(frames: List[Tuple[int, np.ndarray]], percentile: float = 30.0) -> np.ndarray:
    if not frames:
        raise RuntimeError("Sem frames para background")
    arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in frames], axis=0)
    percentile = float(np.clip(percentile, 0.0, 100.0))
    bg = np.percentile(arr, percentile, axis=0).astype(np.uint8)
    return bg


def background_cache_path(video_path: str, p: Params) -> str:
    root = os.path.abspath(video_path)
    return root + f".bg_v10_n{int(p.motion_sample_count)}_p{int(round(float(p.background_percentile)))}.npz"


def load_or_build_background(video_path: str, p: Params, samples: List[Tuple[int, np.ndarray]], meta: Dict[str, object]) -> np.ndarray:
    cache_path = background_cache_path(video_path, p)
    if p.background_cache_enabled and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=False)
            bg = data["bg"]
            ok = (
                int(data["width"]) == int(meta["width"])
                and int(data["height"]) == int(meta["height"])
                and int(data["frame_count"]) == int(meta["frame_count"])
                and int(data["motion_sample_count"]) == int(p.motion_sample_count)
                and abs(float(data["background_percentile"]) - float(p.background_percentile)) < 1e-6
            )
            if ok and bg.ndim == 2:
                return bg.astype(np.uint8)
        except Exception:
            pass

    bg = estimate_background(samples, p.background_percentile)
    if p.background_cache_enabled:
        try:
            np.savez_compressed(
                cache_path,
                bg=bg,
                width=int(meta["width"]),
                height=int(meta["height"]),
                frame_count=int(meta["frame_count"]),
                motion_sample_count=int(p.motion_sample_count),
                background_percentile=float(p.background_percentile),
            )
        except Exception:
            pass
    return bg


def auto_motion_roi(video_path: str, p: Params, samples: Optional[List[Tuple[int, np.ndarray]]] = None, bg: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]:
    if samples is None:
        samples = sample_frames(video_path, p.motion_sample_count)
    if not samples:
        raise RuntimeError("Não consegui amostrar frames")
    H, W = samples[0][1].shape[:2]
    if bg is None:
        bg = estimate_background(samples, p.background_percentile)

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


def detect_candidates_in_frame(frame: np.ndarray, bg_gray: np.ndarray, roi: Tuple[int, int, int, int], template: Optional[np.ndarray], p: Params, gpu: Optional[FrameGpuAccelerator] = None) -> List[dict]:
    x0, y0, w0, h0 = roi
    frame_roi = frame[y0:y0+h0, x0:x0+w0]
    bg_roi = bg_gray[y0:y0+h0, x0:x0+w0]

    # V9: pré-processamento acelerado por GPU quando disponível.
    # As saídas são as mesmas matrizes CPU usadas pela lógica da V8.
    if gpu is not None:
        gray, gray_blur, diff_blur = gpu.preprocess_ball(frame_roi, bg_roi)
    else:
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, bg_roi)
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
    if gpu is not None:
        mask = gpu.morphology(mask, cv2.MORPH_OPEN, k1, iterations=1)
        mask = gpu.morphology(mask, cv2.MORPH_CLOSE, k2, iterations=2)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        cand = candidate_from_contour(c, (x0, y0), gray, diff_blur, template, p)
        if cand is not None:
            candidates.append(cand)

    # Hough como complemento, mas sem dominar
    # Útil quando contorno fragmenta. Na V9, usa CUDA Hough quando disponível
    # e cai para o mesmo Hough CPU da V8 caso contrário.
    try:
        circles = gpu.hough_circles(gray_blur, p) if gpu is not None else None
        if circles is None:
            circles_cpu = cv2.HoughCircles(
                gray_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(20, p.min_ball_radius_px),
                param1=80,
                param2=18,
                minRadius=p.min_ball_radius_px,
                maxRadius=p.max_ball_radius_px,
            )
            circles = np.empty((0, 3), dtype=float) if circles_cpu is None else np.asarray(circles_cpu[0, :], dtype=float)
        if circles is not None and len(circles) > 0:
            for c in np.round(circles).astype(int)[:10]:
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




# ============================================================
# V8 - pós-processamento independente do rastreamento da V6
# ============================================================

def _red_led_raw_score(frame: np.ndarray, p: Params, gpu: Optional[FrameGpuAccelerator] = None) -> dict:
    hsv = gpu.hsv(frame) if gpu is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(frame)

    mask_hsv_1 = cv2.inRange(hsv, (0, 90, 80), (12, 255, 255))
    mask_hsv_2 = cv2.inRange(hsv, (168, 90, 80), (180, 255, 255))

    red_dom = (
        (r.astype(np.int16) > 105)
        & (r.astype(np.int16) > (g.astype(np.int16) + 35))
        & (r.astype(np.int16) > (b.astype(np.int16) + 35))
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_and(cv2.bitwise_or(mask_hsv_1, mask_hsv_2), red_dom)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if gpu is not None:
        mask = gpu.morphology(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = gpu.morphology(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

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
                "sync_red_centroid_x_px": cx,
                "sync_red_centroid_y_px": cy,
                "sync_raw_score": float(score),
            }

    return best


def finalize_sync_dataframe(sync_df: pd.DataFrame, p: Params) -> Tuple[pd.DataFrame, int, int]:
    if sync_df.empty:
        return sync_df, -1, -1

    raw = sync_df["sync_raw_score"].to_numpy(dtype=float)
    if not np.isfinite(raw).any():
        score_norm = np.zeros_like(raw)
    else:
        lo = float(np.nanmedian(raw))
        hi = float(np.nanpercentile(raw, 99.0))
        if hi <= lo:
            hi = float(np.nanmax(raw))
        if hi <= lo:
            score_norm = np.zeros_like(raw)
        else:
            score_norm = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)

    pulse = score_norm >= float(p.sync_score_threshold)

    # Suavização temporal simples para remover piscadas de 1 frame.
    smooth = max(0, int(p.sync_temporal_smooth_frames))
    if smooth > 0 and len(pulse) > 2:
        pulse_int = pulse.astype(np.uint8)
        for _ in range(smooth):
            # fecha buracos curtos
            for i in range(1, len(pulse_int) - 1):
                if pulse_int[i - 1] and pulse_int[i + 1]:
                    pulse_int[i] = 1
            # remove ilhas curtas
            for i in range(1, len(pulse_int) - 1):
                if not pulse_int[i - 1] and not pulse_int[i + 1]:
                    pulse_int[i] = 0
        pulse = pulse_int.astype(bool)

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

    sync_df = sync_df.copy()
    sync_df["sync_score"] = score_norm
    sync_df["sync_pulse"] = pulse.astype(int)
    sync_df["sync_event"] = events
    sync_df["sync_rising_edge"] = rising.astype(int)
    sync_df["sync_falling_edge"] = falling.astype(int)
    sync_df["sync_rising_frame"] = rise_frame
    sync_df["sync_falling_frame"] = fall_frame
    sync_df["sync_pulse_width_frames"] = (fall_frame - rise_frame) if (rise_frame >= 0 and fall_frame >= 0) else np.nan
    return sync_df, rise_frame, fall_frame



def _estimate_ruler_tick_spacing_from_rows(dark_mask: np.ndarray, roi_w: int, p: Params) -> float:
    if dark_mask.size == 0 or roi_w <= 0:
        return float("nan")

    # As marcas da régua ficam concentradas no lado esquerdo da fita.
    best_spacing = float("nan")
    best_count = 0

    for frac in (0.25, 0.35, 0.45, 0.55):
        strip_w = max(8, int(frac * roi_w))
        strip = dark_mask[:, :strip_w]
        row_score = strip.sum(axis=1).astype(float) / 255.0
        if len(row_score) < 10:
            continue

        smooth = np.convolve(row_score, np.ones(3) / 3.0, mode="same")
        thr = max(float(np.percentile(smooth, 85.0)), 3.0)

        candidates = np.where(smooth >= thr)[0]
        peaks = []
        min_dist = max(6, int(round(float(p.ruler_tick_min_spacing_px))))
        for idx in candidates:
            lo = max(0, idx - 2)
            hi = min(len(smooth), idx + 3)
            if smooth[idx] < np.max(smooth[lo:hi]) - 1e-9:
                continue
            if peaks and idx - peaks[-1] < min_dist:
                if smooth[idx] > smooth[peaks[-1]]:
                    peaks[-1] = int(idx)
            else:
                peaks.append(int(idx))

        if len(peaks) < 6:
            continue

        diffs = np.diff(np.asarray(peaks, dtype=float))
        # No vídeo de bancada, as menores distâncias reais relevantes são as marcas de 1 mm.
        useful = diffs[(diffs >= 7.0) & (diffs <= 18.0)]
        if len(useful) < 3:
            useful = diffs[
                (diffs >= float(p.ruler_tick_min_spacing_px))
                & (diffs <= float(p.ruler_tick_max_spacing_px))
            ]
        if len(useful) < 3:
            continue

        bins = np.arange(max(4.0, useful.min() - 1.0), min(30.0, useful.max() + 1.5), 0.5)
        if len(bins) < 3:
            spacing = float(np.median(useful))
            count = len(useful)
        else:
            hist, edges = np.histogram(useful, bins=bins)
            best = int(np.argmax(hist))
            selected = useful[(useful >= edges[best]) & (useful < edges[best + 1])]
            spacing = float(np.median(selected)) if len(selected) else float(np.median(useful))
            count = int(hist[best])

        if count > best_count:
            best_count = count
            best_spacing = spacing

    return best_spacing

def detect_auto_ruler_from_frame(frame: np.ndarray, p: Params) -> dict:
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    x_search0 = int(W * (1.0 - float(p.ruler_search_right_ratio)))
    x_search0 = max(0, min(W - 2, x_search0))

    white = ((val > 125) & (sat < 110)).astype(np.uint8) * 255
    white[:, :x_search0] = 0

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 15))
    white2 = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(white2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if h > 0.25 * H and w > 20 and area > 0.01 * W * H:
            boxes.append((x, y, w, h, area))

    if boxes:
        # Preferir região alta, vertical e à direita.
        boxes.sort(key=lambda b: b[4] * (1.0 + b[0] / max(1, W)), reverse=True)
        x, y, w, h, _ = boxes[0]
        pad = 8
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(W - x, w + 2 * pad)
        h = min(H - y, h + 2 * pad)
    else:
        x = int(W * 0.58)
        y = int(H * 0.03)
        w = int(W * 0.24)
        h = int(H * 0.82)

    roi_gray = gray[y:y+h, x:x+w]
    if roi_gray.size == 0:
        return {"auto_ruler_ok": False, "auto_ruler_reason": "empty_roi"}

    blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    dark = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )
    # Mantém linhas horizontais curtas da escala.
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dark_h = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel_h, iterations=1)
    row_spacing_px = _estimate_ruler_tick_spacing_from_rows(dark_h, w, p)

    lines = cv2.HoughLinesP(
        dark_h, 1, np.pi / 180.0,
        threshold=8,
        minLineLength=5,
        maxLineGap=3,
    )

    tick_pts = []
    tick_lengths = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 5:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle > 25 and angle < 155:
                continue
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            # Ticks ficam preferencialmente na metade esquerda da régua.
            if cx > 0.62 * w:
                continue
            tick_pts.append([cx + x, cy + y])
            tick_lengths.append(length)

    if len(tick_pts) < 8:
        # Fallback por componentes conectados.
        num, labels, stats, cent = cv2.connectedComponentsWithStats(dark_h, 8)
        for i in range(1, num):
            xx, yy, ww, hh, area = stats[i]
            if area < 5 or area > 800:
                continue
            if ww < 5 or hh > 12:
                continue
            if ww / max(1, hh) < 1.4:
                continue
            cx, cy = cent[i]
            if cx > 0.62 * w:
                continue
            tick_pts.append([float(cx + x), float(cy + y)])
            tick_lengths.append(float(ww))

    if len(tick_pts) < 8:
        return {
            "auto_ruler_ok": False,
            "auto_ruler_reason": "few_ticks",
            "ruler_roi_x_px": x, "ruler_roi_y_px": y,
            "ruler_roi_w_px": w, "ruler_roi_h_px": h,
        }

    pts = np.asarray(tick_pts, dtype=float)

    # Estimar direção da régua por PCA nos pontos dos ticks.
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    u = vh[0]
    if u[1] < 0:
        u = -u
    # Evita eixo horizontal em casos ruins.
    if abs(u[1]) < 0.55:
        u = np.array([0.0, 1.0], dtype=float)

    nvec = np.array([-u[1], u[0]], dtype=float)
    s = pts @ u
    q = pts @ nvec

    # Agrupa múltiplas detecções da mesma marca.
    order = np.argsort(s)
    s_sorted = s[order]
    pts_sorted = pts[order]
    clusters = []
    current = [pts_sorted[0]]
    current_s = [s_sorted[0]]
    for ss, pp in zip(s_sorted[1:], pts_sorted[1:]):
        if abs(ss - np.mean(current_s)) <= 3.0:
            current.append(pp)
            current_s.append(ss)
        else:
            clusters.append((float(np.mean(current_s)), np.mean(current, axis=0)))
            current = [pp]
            current_s = [ss]
    clusters.append((float(np.mean(current_s)), np.mean(current, axis=0)))

    s_clusters = np.array([c[0] for c in clusters], dtype=float)
    pts_clusters = np.array([c[1] for c in clusters], dtype=float)

    if len(s_clusters) < 8:
        return {
            "auto_ruler_ok": False,
            "auto_ruler_reason": "few_tick_clusters",
            "ruler_roi_x_px": x, "ruler_roi_y_px": y,
            "ruler_roi_w_px": w, "ruler_roi_h_px": h,
        }

    diffs = np.diff(np.sort(s_clusters))
    diffs = diffs[
        (diffs >= float(p.ruler_tick_min_spacing_px))
        & (diffs <= float(p.ruler_tick_max_spacing_px))
    ]

    spacing_from_clusters = float("nan")
    spacing_hist_best_count = 0
    spacing_hist_total = max(1, len(diffs))
    if len(diffs) >= 4:
        bin_w = 0.5
        bins = np.arange(float(p.ruler_tick_min_spacing_px), float(p.ruler_tick_max_spacing_px) + bin_w, bin_w)
        hist, edges = np.histogram(diffs, bins=bins)
        best = int(np.argmax(hist))
        spacing_hist_best_count = int(hist[best])
        spacing_candidates = diffs[(diffs >= edges[best]) & (diffs < edges[best + 1])]
        spacing_from_clusters = float(np.median(spacing_candidates)) if len(spacing_candidates) else float(np.median(diffs))

    # Preferir o espaçamento por projeção de linhas horizontais; ele é mais estável
    # contra duplicatas do Hough e letras/números da régua.
    if np.isfinite(row_spacing_px) and row_spacing_px > 0:
        spacing_px = float(row_spacing_px) / max(abs(float(u[1])), 0.25)
    elif np.isfinite(spacing_from_clusters) and spacing_from_clusters > 0:
        spacing_px = spacing_from_clusters
    else:
        return {
            "auto_ruler_ok": False,
            "auto_ruler_reason": "no_spacing",
            "ruler_roi_x_px": x, "ruler_roi_y_px": y,
            "ruler_roi_w_px": w, "ruler_roi_h_px": h,
        }

    mm_per_px_ruler = 1.0 / spacing_px if spacing_px > 0 else np.nan

    smin = float(np.min(s_clusters))
    smax = float(np.max(s_clusters))
    qmed = float(np.median(pts_clusters @ nvec))
    top = u * smin + nvec * qmed
    bottom = u * smax + nvec * qmed

    return {
        "auto_ruler_ok": True,
        "auto_ruler_reason": "OK",
        "ruler_top_x_px": float(top[0]),
        "ruler_top_y_px": float(top[1]),
        "ruler_bottom_x_px": float(bottom[0]),
        "ruler_bottom_y_px": float(bottom[1]),
        "ruler_u_x": float(u[0]),
        "ruler_u_y": float(u[1]),
        "ruler_n_x": float(nvec[0]),
        "ruler_n_y": float(nvec[1]),
        "ruler_origin_x_px": float(top[0]),
        "ruler_origin_y_px": float(top[1]),
        "ruler_dx_px": float(bottom[0] - top[0]),
        "ruler_dy_px": float(bottom[1] - top[1]),
        "ruler_len_px": float(math.hypot(bottom[0] - top[0], bottom[1] - top[1])),
        "ruler_angle_deg": float(math.degrees(math.atan2(u[0], u[1]))),
        "ruler_tick_spacing_px": spacing_px,
        "ruler_tick_spacing_row_px": float(row_spacing_px) if np.isfinite(row_spacing_px) else np.nan,
        "ruler_tick_spacing_cluster_px": float(spacing_from_clusters) if np.isfinite(spacing_from_clusters) else np.nan,
        "mm_per_px_ruler": mm_per_px_ruler,
        "px_per_mm_ruler": spacing_px,
        "ruler_roi_x_px": x,
        "ruler_roi_y_px": y,
        "ruler_roi_w_px": w,
        "ruler_roi_h_px": h,
        "ruler_tick_count": int(len(s_clusters)),
        "ruler_confidence": float(min(1.0, len(s_clusters) / 25.0) * min(1.0, spacing_hist_best_count / max(1.0, spacing_hist_total * 0.35))),
        "ruler_tick_points": pts_clusters.tolist(),
    }


def apply_auto_ruler_correction(df: pd.DataFrame, ruler: dict, reference_frame: Optional[int] = None) -> pd.DataFrame:
    df = df.copy()

    base_cols = {
        "auto_ruler_ok": False,
        "auto_ruler_reason": ruler.get("auto_ruler_reason", "not_run"),
        "ruler_top_x_px": np.nan,
        "ruler_top_y_px": np.nan,
        "ruler_bottom_x_px": np.nan,
        "ruler_bottom_y_px": np.nan,
        "ruler_dx_px": np.nan,
        "ruler_dy_px": np.nan,
        "ruler_len_px": np.nan,
        "ruler_angle_deg": np.nan,
        "ruler_tick_spacing_px": np.nan,
        "mm_per_px_ruler": np.nan,
        "px_per_mm_ruler": np.nan,
        "ruler_confidence": np.nan,
        "y_top_fixed_ruler_mm_corr": np.nan,
        "y_center_ruler_mm_corr": np.nan,
        "z_top_fixed_mm_rel_corr": np.nan,
        "z_center_mm_rel_corr": np.nan,
    }
    for k, v in base_cols.items():
        df[k] = v

    if not ruler.get("auto_ruler_ok", False):
        return df

    origin = np.array([ruler["ruler_origin_x_px"], ruler["ruler_origin_y_px"]], dtype=float)
    u = np.array([ruler["ruler_u_x"], ruler["ruler_u_y"]], dtype=float)
    mm_per_px = float(ruler["mm_per_px_ruler"])

    top_pts = np.column_stack([
        df["x_center_px"].to_numpy(dtype=float),
        df["y_top_fixed_px"].to_numpy(dtype=float),
    ])
    center_pts = np.column_stack([
        df["x_center_px"].to_numpy(dtype=float),
        df["y_center_px"].to_numpy(dtype=float),
    ])

    s_top = (top_pts - origin) @ u
    s_center = (center_pts - origin) @ u

    df["y_top_fixed_ruler_mm_corr"] = s_top * mm_per_px
    df["y_center_ruler_mm_corr"] = s_center * mm_per_px

    valid_mask = df["valid"].astype(bool) & df["y_top_fixed_ruler_mm_corr"].notna()
    ref_idx = None
    if reference_frame is not None and reference_frame >= 0:
        candidates = df.index[valid_mask & (df["frame"] >= int(reference_frame))]
        if len(candidates):
            ref_idx = candidates[0]
    if ref_idx is None:
        candidates = df.index[valid_mask]
        if len(candidates):
            ref_idx = candidates[0]

    if ref_idx is not None:
        ref_top = float(df.loc[ref_idx, "y_top_fixed_ruler_mm_corr"])
        ref_center = float(df.loc[ref_idx, "y_center_ruler_mm_corr"])
        # u aponta para baixo na régua; movimento para cima reduz s.
        df["z_top_fixed_mm_rel_corr"] = ref_top - df["y_top_fixed_ruler_mm_corr"]
        df["z_center_mm_rel_corr"] = ref_center - df["y_center_ruler_mm_corr"]

    for k, v in ruler.items():
        if k == "ruler_tick_points":
            continue
        if isinstance(v, (str, bool, int, float, np.floating)):
            df[k] = v

    return df


def add_time_and_sync_columns(df: pd.DataFrame, meta: Dict[str, object], sync_df: pd.DataFrame, p: Params) -> Tuple[pd.DataFrame, int, int]:
    df = df.copy()
    fps_playback = float(meta.get("fps_playback", meta.get("fps", 0.0)) or 0.0)
    fps_capture = float(meta.get("fps_capture", fps_playback) or 0.0)
    confidence = str(meta.get("metadata_time_confidence", "LOW"))
    source = str(meta.get("metadata_time_source", "unknown"))

    if not sync_df.empty:
        df = df.merge(sync_df, on="frame", how="left")
    else:
        for col in [
            "sync_red_area_px", "sync_red_centroid_x_px", "sync_red_centroid_y_px",
            "sync_raw_score", "sync_score", "sync_pulse", "sync_event",
            "sync_rising_edge", "sync_falling_edge", "sync_rising_frame",
            "sync_falling_frame", "sync_pulse_width_frames",
        ]:
            df[col] = np.nan

    rise_frame = int(df["sync_rising_frame"].dropna().iloc[0]) if df["sync_rising_frame"].notna().any() else -1
    fall_frame = int(df["sync_falling_frame"].dropna().iloc[0]) if df["sync_falling_frame"].notna().any() else -1

    # Se o metadado não trouxe FPS de captura confiável, usa a largura do pulso
    # conhecido do Arduino como estimativa explícita e marcada.
    if confidence == "LOW" and rise_frame >= 0 and fall_frame > rise_frame and p.sync_known_pulse_width_ms > 0:
        pulse_frames = fall_frame - rise_frame
        fps_from_pulse = 1000.0 * float(pulse_frames) / float(p.sync_known_pulse_width_ms)
        if fps_from_pulse > 1:
            fps_capture = fps_from_pulse
            confidence = "PULSE_ESTIMATED"
            source = "sync_led_known_width"

    df["time_video_ms"] = (df["frame"] / fps_playback * 1000.0) if fps_playback else np.nan
    df["time_real_ms"] = (df["frame"] / fps_capture * 1000.0) if fps_capture else np.nan

    if rise_frame >= 0 and fps_capture:
        df["time_sync_ms"] = (df["frame"] - rise_frame) / fps_capture * 1000.0
        df["time_from_sync_rise_ms"] = df["time_sync_ms"]
    else:
        df["time_sync_ms"] = np.nan
        df["time_from_sync_rise_ms"] = np.nan

    if fall_frame >= 0 and fps_capture:
        df["time_from_sync_fall_ms"] = (df["frame"] - fall_frame) / fps_capture * 1000.0
    else:
        df["time_from_sync_fall_ms"] = np.nan

    if rise_frame >= 0 and fall_frame >= 0 and fps_capture:
        width_ms = (fall_frame - rise_frame) / fps_capture * 1000.0
    else:
        width_ms = np.nan
    df["sync_pulse_width_ms"] = width_ms

    df["fps_playback"] = fps_playback
    df["fps_capture"] = fps_capture
    df["time_scale_detected"] = (fps_playback / fps_capture) if fps_capture else np.nan
    df["metadata_time_source"] = source
    df["metadata_time_confidence"] = confidence
    df["frame_sync"] = df["frame"] - rise_frame if rise_frame >= 0 else np.nan

    return df, rise_frame, fall_frame


def crop_to_synced_section(df: pd.DataFrame, rise_frame: int, p: Params) -> pd.DataFrame:
    if not p.sync_crop_to_rising_edge or rise_frame < 0:
        return df.copy()
    return df.loc[df["frame"] >= rise_frame].copy()


def make_sync_preview(video_path: str, df: pd.DataFrame, out_path: str, rise_frame: int, fall_frame: int):
    if rise_frame < 0 and fall_frame < 0:
        return

    meta = read_video_metadata(video_path)
    n = int(meta["frame_count"])
    frames = []
    for f in [rise_frame - 2, rise_frame - 1, rise_frame, rise_frame + 1, fall_frame - 1, fall_frame, fall_frame + 1]:
        if f >= 0 and f < n and f not in frames:
            frames.append(f)

    cap = cv2.VideoCapture(video_path)
    thumbs = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()
        row = df.loc[df["frame"] == f]
        if len(row):
            r = row.iloc[0]
            if np.isfinite(r.get("sync_red_centroid_x_px", np.nan)):
                cx = int(round(r["sync_red_centroid_x_px"]))
                cy = int(round(r["sync_red_centroid_y_px"]))
                cv2.circle(vis, (cx, cy), 18, (0, 255, 255), 2)
            txt = f"f={f} {r.get('sync_event','')} score={float(r.get('sync_score',0)):.2f}"
        else:
            txt = f"f={f}"
        cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
        thumb_w = 420
        scale = thumb_w / vis.shape[1]
        thumbs.append(cv2.resize(vis, (thumb_w, int(vis.shape[0] * scale))))
    cap.release()

    if not thumbs:
        return

    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    th, tw = thumbs[0].shape[:2]
    canvas = np.full((rows * th, cols * tw, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        rr = i // cols
        cc = i % cols
        canvas[rr*th:rr*th+thumb.shape[0], cc*tw:cc*tw+thumb.shape[1]] = thumb
    cv2.imwrite(out_path, canvas)


def make_ruler_preview(video_path: str, ruler: dict, out_path: str):
    if not ruler:
        return
    meta = read_video_metadata(video_path)
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    vis = frame.copy()

    x = int(ruler.get("ruler_roi_x_px", 0))
    y = int(ruler.get("ruler_roi_y_px", 0))
    w = int(ruler.get("ruler_roi_w_px", 0))
    h = int(ruler.get("ruler_roi_h_px", 0))
    if w > 0 and h > 0:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 180, 0), 2)

    if ruler.get("auto_ruler_ok", False):
        pt1 = (int(round(ruler["ruler_top_x_px"])), int(round(ruler["ruler_top_y_px"])))
        pt2 = (int(round(ruler["ruler_bottom_x_px"])), int(round(ruler["ruler_bottom_y_px"])))
        cv2.line(vis, pt1, pt2, (0, 255, 255), 2)
        pts = ruler.get("ruler_tick_points", [])[:120]
        for pnt in pts:
            cv2.circle(vis, (int(round(pnt[0])), int(round(pnt[1]))), 2, (0, 0, 255), -1)
        txt = f"ruler auto OK | {ruler['px_per_mm_ruler']:.2f} px/mm | angle={ruler['ruler_angle_deg']:.2f} deg"
    else:
        txt = f"ruler auto FAIL: {ruler.get('auto_ruler_reason','unknown')}"

    cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def process_video(
    video_path: str,
    out_csv: str,
    p: Params,
    preview: Optional[str] = None,
    plot: Optional[str] = None,
    debug_candidates: Optional[str] = None,
    full_csv: Optional[str] = None,
    sync_preview: Optional[str] = None,
    ruler_preview: Optional[str] = None,
):
    meta = read_video_metadata(video_path)
    fps = float(meta["fps"])
    W, H = int(meta["width"]), int(meta["height"])
    n = int(meta["frame_count"])

    gpu = FrameGpuAccelerator(p)

    samples = sample_frames(video_path, p.motion_sample_count)
    bg_gray = load_or_build_background(video_path, p, samples, meta)

    if p.auto_roi or p.roi is None:
        roi = auto_motion_roi(video_path, p, samples=samples, bg=bg_gray)
    else:
        roi = clamp_roi(p.roi, W, H, p.frame_border_ignore_px)

    ruler = {}
    if p.auto_ruler:
        # Usa o background/mediana para reduzir interferência da bolinha.
        ruler_frame = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
        ruler = detect_auto_ruler_from_frame(ruler_frame, p)
    else:
        ruler = {"auto_ruler_ok": False, "auto_ruler_reason": "disabled"}

    roi = restrict_roi_by_ruler(roi, ruler, W, H, p)

    all_candidates: List[List[dict]] = []
    candidate_rows = []
    sync_rows = []

    template = None

    # Primeiro passe: mantém o rastreamento da V6 intacto e coleta o LED em paralelo.
    for frame_idx, frame in iter_video_frames(video_path, p):
        cands = detect_candidates_in_frame(frame, bg_gray, roi, template, p, gpu)

        if cands:
            template = maybe_update_template(frame, cands[0], template, p)

        all_candidates.append(cands)

        if p.sync_enabled:
            srow = {"frame": frame_idx}
            srow.update(_red_led_raw_score(frame, p, gpu))
            sync_rows.append(srow)

        if debug_candidates:
            for rank, c in enumerate(cands):
                row = {"frame": frame_idx, "rank": rank}
                row.update(c)
                candidate_rows.append(row)


    track = choose_track_dp(all_candidates, p)

    rows = []
    for i, cand in enumerate(track):
        time_video_s = i / fps if fps else np.nan
        # Mantém colunas legadas da V6; as colunas oficiais da V8 ficam em ms.
        time_real_s = time_video_s
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
    # Raio fixo da V6 para reduzir ruído do topo
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

    sync_df = pd.DataFrame(sync_rows) if sync_rows else pd.DataFrame({"frame": df["frame"]})
    sync_df, rise_frame, fall_frame = finalize_sync_dataframe(sync_df, p)

    df, rise_frame, fall_frame = add_time_and_sync_columns(df, meta, sync_df, p)
    df = apply_auto_ruler_correction(df, ruler, reference_frame=rise_frame if rise_frame >= 0 else None)

    # Compatibilidade: se a régua automática falhar, usa escala manual antiga caso exista.
    if not bool(df["auto_ruler_ok"].iloc[0]) if len(df) else True:
        mm_per_px_effective = 0.0
        if p.mm_per_px and p.mm_per_px > 0:
            mm_per_px_effective = float(p.mm_per_px)
        elif p.px_per_mm and p.px_per_mm > 0:
            mm_per_px_effective = 1.0 / float(p.px_per_mm)
        if mm_per_px_effective > 0:
            df["mm_per_px"] = mm_per_px_effective
            first_valid_idx = df.index[df["valid"].astype(bool) & df["y_top_fixed_px"].notna()]
            if len(first_valid_idx) > 0:
                ref_y_top = float(df.loc[first_valid_idx[0], "y_top_fixed_px"])
                ref_y_center = float(df.loc[first_valid_idx[0], "y_center_px"])
                df["z_top_fixed_mm_rel"] = (ref_y_top - df["y_top_fixed_px"]) * mm_per_px_effective
                df["z_center_mm_rel"] = (ref_y_center - df["y_center_px"]) * mm_per_px_effective
                df["z_top_fixed_mm_rel_corr"] = df["z_top_fixed_mm_rel"]
                df["z_center_mm_rel_corr"] = df["z_center_mm_rel"]

    # Suavização opcional em colunas extras, não substitui bruto.
    if p.median_smooth_window and p.median_smooth_window > 1:
        win = int(p.median_smooth_window)
        if win % 2 == 0:
            win += 1
        for col in [
            "x_center_px", "y_center_px", "radius_px", "y_top_px", "y_top_fixed_px",
            "z_top_fixed_mm_rel_corr", "z_center_mm_rel_corr"
        ]:
            if col in df.columns:
                df[col + "_median"] = df[col].rolling(win, center=True, min_periods=1).median()

    df = flag_physical_outliers(df, p)

    # atributos úteis
    df["roi_x_px"] = roi[0]
    df["roi_y_px"] = roi[1]
    df["roi_w_px"] = roi[2]
    df["roi_h_px"] = roi[3]
    df["video_fps"] = fps
    df["time_scale"] = df["time_scale_detected"]
    df["gpu_enabled"] = bool(gpu.enabled)
    df["gpu_status"] = gpu.reason
    df["gpu_fallback_count"] = int(getattr(gpu, "gpu_fallback_count", 0))
    df["background_cache_enabled"] = bool(p.background_cache_enabled)
    df["background_percentile"] = float(p.background_percentile)

    if full_csv:
        df.to_csv(full_csv, index=False)

    treated = crop_to_synced_section(df, rise_frame, p)
    treated.to_csv(out_csv, index=False)

    if debug_candidates:
        pd.DataFrame(candidate_rows).to_csv(debug_candidates, index=False)

    if preview:
        make_preview(video_path, df, roi, preview, p)

    if sync_preview:
        make_sync_preview(video_path, df, sync_preview, rise_frame, fall_frame)

    if ruler_preview:
        make_ruler_preview(video_path, ruler, ruler_preview)

    if plot:
        make_plot(treated, plot)

    print("MAGLEV VIDEO TRACKER V10")
    print(f"Video: {video_path}")
    print(f"Resolution: {W}x{H}")
    print(f"FPS playback: {float(df['fps_playback'].iloc[0]) if len(df) else float('nan'):.4f}")
    print(f"FPS capture:  {float(df['fps_capture'].iloc[0]) if len(df) else float('nan'):.4f}")
    print(f"Time source:  {str(df['metadata_time_source'].iloc[0]) if len(df) else 'unknown'}")
    print(f"Time conf:    {str(df['metadata_time_confidence'].iloc[0]) if len(df) else 'unknown'}")
    print(f"Frames: {n}")
    print(f"GPU: {'ON' if gpu.enabled else 'OFF'} | {gpu.reason} | fallbacks={getattr(gpu, 'gpu_fallback_count', 0)}")
    print(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print(f"Background: percentile={p.background_percentile:.1f} | cache={'ON' if p.background_cache_enabled else 'OFF'}")
    print(f"Rising frame: {rise_frame}")
    print(f"Falling frame: {fall_frame}")
    if len(df) and df["sync_pulse_width_ms"].notna().any():
        print(f"Sync pulse width ms: {float(df['sync_pulse_width_ms'].dropna().iloc[0]):.2f}")
    if ruler.get("auto_ruler_ok", False):
        print(f"Auto ruler: OK | px/mm={ruler['px_per_mm_ruler']:.4f} | mm/px={ruler['mm_per_px_ruler']:.6f} | angle={ruler['ruler_angle_deg']:.2f} deg")
    else:
        print(f"Auto ruler: FAIL | {ruler.get('auto_ruler_reason', 'unknown')}")
    print(f"Output CSV treated: {out_csv}")
    if full_csv:
        print(f"Output CSV full:    {full_csv}")
    print(f"Valid frames full: {int(df['valid'].sum())}/{len(df)}")
    print(f"Valid frames treated: {int(treated['valid'].sum())}/{len(treated)}")
    if df["y_center_px"].notna().any():
        print(f"y_center_px range: {df['y_center_px'].min():.2f} .. {df['y_center_px'].max():.2f}")
        print(f"y_top_px range:    {df['y_top_px'].min():.2f} .. {df['y_top_px'].max():.2f}")
        print(f"confidence range:  {df['confidence'].min():.3f} .. {df['confidence'].max():.3f}")
        if "fixed_radius_px" in df.columns and df["fixed_radius_px"].notna().any():
            print(f"fixed_radius_px:   {df['fixed_radius_px'].dropna().iloc[0]:.2f}")


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

    if "time_sync_ms" in df.columns and df["time_sync_ms"].notna().any():
        x = df["time_sync_ms"]
        xlabel = "time_sync_ms (ms)"
    elif "time_real_ms" in df.columns and df["time_real_ms"].notna().any():
        x = df["time_real_ms"]
        xlabel = "time_real_ms (ms)"
    else:
        x = df["time_real_s"]
        xlabel = "tempo real estimado (s)"

    if "valid_signal" in df.columns:
        valid = df["valid_signal"].astype(bool)
    elif "valid" in df.columns:
        valid = df["valid"].astype(bool)
    else:
        valid = pd.Series(True, index=df.index)

    y_candidates = [
        ("z_top_fixed_mm_rel_corr_median", "z_top_fixed_mm_rel_corr smooth"),
        ("z_top_fixed_mm_rel_corr", "z_top_fixed_mm_rel_corr raw"),
        ("z_top_fixed_mm_rel_median", "z_top_fixed_mm_rel smooth"),
        ("z_top_fixed_mm_rel", "z_top_fixed_mm_rel raw"),
        ("y_top_fixed_px_median", "y_top_fixed_px smooth"),
        ("y_top_fixed_px", "y_top_fixed_px raw"),
        ("y_center_px_median", "y_center_px smooth"),
        ("y_center_px", "y_center_px raw"),
    ]
    plotted = False
    for col, label in y_candidates:
        if col in df.columns and df[col].notna().any():
            y = df[col].where(valid)
            ax.plot(x, y, label=label, linewidth=1.5)
            plotted = True
            break

    if "z_center_mm_rel_corr" in df.columns and df["z_center_mm_rel_corr"].notna().any():
        center_col = "z_center_mm_rel_corr_median" if "z_center_mm_rel_corr_median" in df.columns else "z_center_mm_rel_corr"
        ax.plot(x, df[center_col].where(valid), label=center_col, linewidth=0.9, alpha=0.65)

    if "sync_falling_edge" in df.columns and "time_from_sync_fall_ms" in df.columns:
        fall_rows = df.index[df["sync_falling_edge"].fillna(0).astype(int) == 1]
        if len(fall_rows):
            fall_t = float(df.loc[fall_rows[0], "time_sync_ms"])
            ax.axvline(fall_t, linestyle=":", linewidth=1.0, label="FALLING_EDGE")

    if "sync_rising_edge" in df.columns:
        ax.axvline(0.0, linestyle="--", linewidth=1.0, label="RISING_EDGE")

    if plotted:
        ax.set_ylabel("deslocamento corrigido (mm) / posição")
    else:
        ax.set_ylabel("posição")

    ax.set_xlabel(xlabel)
    ax.set_title("Maglev video tracking V10")
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
    ap.add_argument("--out", default=None, help="CSV tratado de saída. Se omitido, salva em <output-dir>/tracking_tratado.csv")
    ap.add_argument("--full-out", default=None, help="CSV completo. Se omitido, salva em <output-dir>/tracking_full.csv")
    ap.add_argument("--output-dir", default=None, help="Pasta de saída. Se omitida, usa <nome_do_video>/")
    ap.add_argument("--preview", default=None, help="Imagem JPG/PNG com frames anotados. Se omitido, salva em <output-dir>/preview.jpg")
    ap.add_argument("--plot", default=None, help="Gráfico PNG da trajetória tratada")
    ap.add_argument("--sync-preview", default=None, help="Preview do LED de sincronismo")
    ap.add_argument("--ruler-preview", default=None, help="Preview da régua detectada automaticamente")
    ap.add_argument("--debug-candidates", default=None, help="CSV opcional com candidatos por frame")
    ap.add_argument("--config", default=None, help="JSON de parâmetros")
    ap.add_argument("--write-default-config", default=None, help="Escreve JSON default e sai")
    ap.add_argument("--roi", default=None, help="ROI manual x,y,w,h. Se usado, desliga auto_roi.")
    ap.add_argument("--median-smooth-window", type=int, default=None, help="Janela mediana opcional para colunas extras")
    ap.add_argument("--fixed-radius-px", type=float, default=None, help="Raio fixo da bolinha em pixels")
    ap.add_argument("--ball-diameter-mm", type=float, default=None, help="Diâmetro real da bolinha em mm")
    ap.add_argument("--px-per-mm", type=float, default=None, help="Fallback manual: pixels por mm")
    ap.add_argument("--mm-per-px", type=float, default=None, help="Fallback manual: mm por pixel")
    ap.add_argument("--no-auto-fixed-radius", action="store_true", help="Desliga cálculo automático do raio fixo pela mediana")
    ap.add_argument("--no-auto-ruler", action="store_true", help="Desliga calibração automática da régua")
    ap.add_argument("--no-sync-crop", action="store_true", help="Não corta o CSV tratado na borda de subida do LED")
    ap.add_argument("--sync-known-pulse-width-ms", type=float, default=None, help="Duração conhecida do LED para fallback temporal; padrão 200 ms")
    ap.add_argument("--no-background-cache", action="store_true", help="Desliga cache de background da V10")
    ap.add_argument("--background-percentile", type=float, default=None, help="Percentil usado para estimar o background; padrão 30")
    ap.add_argument("--no-frame-reader-thread", action="store_true", help="Desliga leitura assíncrona de frames")
    ap.add_argument("--no-auto-exclude-ruler-from-roi", action="store_true", help="Não restringe a ROI pelo início da régua")
    ap.add_argument("--max-physical-delta-mm-per-frame", type=float, default=None, help="Delta máximo por frame para marcar outlier físico")
    ap.add_argument("--no-physical-outlier-filter", action="store_true", help="Desliga marcação de outliers físicos")
    ap.add_argument("--no-gpu", action="store_true", help="Desliga aceleração por GPU e força o caminho CPU da V8")
    ap.add_argument("--gpu-device", type=int, default=None, help="Índice da GPU CUDA a usar; padrão 0")
    args = ap.parse_args()

    if args.write_default_config:
        save_default_config(args.write_default_config)
        print(f"Config default salva em: {args.write_default_config}")
        return

    p = load_config(args.config) if args.config else Params()

    if args.roi:
        p.roi = parse_tuple4(args.roi)
        p.auto_roi = False
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
    if args.no_auto_fixed_radius:
        p.auto_fixed_radius = False
    if args.no_auto_ruler:
        p.auto_ruler = False
    if args.no_sync_crop:
        p.sync_crop_to_rising_edge = False
    if args.sync_known_pulse_width_ms is not None:
        p.sync_known_pulse_width_ms = args.sync_known_pulse_width_ms
    if args.no_background_cache:
        p.background_cache_enabled = False
    if args.background_percentile is not None:
        p.background_percentile = float(args.background_percentile)
    if args.no_frame_reader_thread:
        p.frame_reader_thread = False
    if args.no_auto_exclude_ruler_from_roi:
        p.auto_exclude_ruler_from_roi = False
    if args.max_physical_delta_mm_per_frame is not None:
        p.max_physical_delta_mm_per_frame = float(args.max_physical_delta_mm_per_frame)
    if args.no_physical_outlier_filter:
        p.physical_outlier_enabled = False
    if args.no_gpu:
        p.use_gpu = False
    if args.gpu_device is not None:
        p.gpu_device_id = int(args.gpu_device)

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = args.output_dir if args.output_dir else video_stem
    os.makedirs(output_dir, exist_ok=True)

    out_csv = args.out if args.out else os.path.join(output_dir, "tracking_tratado.csv")
    full_csv = args.full_out if args.full_out else os.path.join(output_dir, "tracking_full.csv")
    preview = args.preview if args.preview else os.path.join(output_dir, "preview.jpg")
    plot = args.plot if args.plot else os.path.join(output_dir, "plot.png")
    sync_preview = args.sync_preview if args.sync_preview else os.path.join(output_dir, "sync_preview.jpg")
    ruler_preview = args.ruler_preview if args.ruler_preview else os.path.join(output_dir, "ruler_preview.jpg")
    debug_candidates = args.debug_candidates if args.debug_candidates else os.path.join(output_dir, "candidates.csv")

    process_video(
        video_path=args.video,
        out_csv=out_csv,
        full_csv=full_csv,
        p=p,
        preview=preview,
        plot=plot,
        sync_preview=sync_preview,
        ruler_preview=ruler_preview,
        debug_candidates=debug_candidates,
    )

    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
