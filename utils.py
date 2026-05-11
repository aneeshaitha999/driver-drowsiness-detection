"""Utility helpers for drowsiness detection system."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


@dataclass
class RuntimeConfig:
    """Runtime thresholds and timing parameters."""

    ear_open_threshold: float = 0.25
    ear_closed_threshold: float = 0.20
    mar_yawn_threshold: float = 0.60
    eyes_closed_seconds_alert: float = 2.0
    distracted_seconds_alert: float = 2.0
    head_pitch_threshold_deg: float = 18.0
    risk_alert_threshold: float = 70.0
    blink_min_duration_s: float = 0.08
    blink_max_duration_s: float = 0.8


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def ratio_from_pairs(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray, p5: np.ndarray, p6: np.ndarray
) -> float:
    """Generic ratio helper used by EAR/MAR-like formulas."""
    denom = 2.0 * euclidean_distance(p1, p4)
    if denom < 1e-6:
        return 0.0
    return (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / denom


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def draw_status_panel(
    frame: np.ndarray,
    status_lines: Iterable[Tuple[str, Tuple[int, int, int]]],
    origin: Tuple[int, int] = (10, 25),
) -> np.ndarray:
    x, y = origin
    for text, color in status_lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        y += 28
    return frame


def overlay_warning(frame: np.ndarray, text: str = "Driver Drowsiness Detected") -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 255), -1)
    cv2.putText(frame, text, (15, 45), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


class CSVEventLogger:
    """CSV logger for continuous metrics and events."""

    def __init__(self, logs_dir: str | Path = "logs"):
        ensure_dir(logs_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = Path(logs_dir) / f"drowsiness_log_{stamp}.csv"
        self._headers = [
            "timestamp",
            "ear",
            "mar",
            "pitch_deg",
            "risk_score",
            "blink_count",
            "yawn_count",
            "status",
            "event",
        ]
        with self.filepath.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(self._headers)

    def log(
        self,
        ear: float,
        mar: float,
        pitch_deg: float,
        risk_score: float,
        blink_count: int,
        yawn_count: int,
        status: str,
        event: str = "",
    ) -> None:
        row = [
            datetime.now().isoformat(timespec="seconds"),
            round(ear, 4),
            round(mar, 4),
            round(pitch_deg, 3),
            round(risk_score, 2),
            blink_count,
            yawn_count,
            status,
            event,
        ]
        with self.filepath.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(row)


def safe_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between vectors in degrees."""
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-9:
        return 0.0
    cos_val = float(np.dot(v1, v2) / norm)
    cos_val = clip(cos_val, -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def format_fps(previous_ts: float, current_ts: float) -> float:
    dt = current_ts - previous_ts
    if dt <= 1e-6:
        return 0.0
    return 1.0 / dt

