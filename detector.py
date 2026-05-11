"""Core face-landmark-based drowsiness detection logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from utils import RuntimeConfig, ratio_from_pairs


@dataclass
class DetectionState:
    ear: float = 0.0
    mar: float = 0.0
    pitch_deg: float = 0.0
    blink_count: int = 0
    yawn_count: int = 0
    risk_score: float = 0.0
    eyes_closed_s: float = 0.0
    head_down_s: float = 0.0
    is_alert: bool = False
    status: str = "ATTENTIVE"
    event: str = ""


class DrowsinessDetector:
    """Analyzes frame landmarks and computes drowsiness risk metrics."""

    # MediaPipe landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]   # p1,p2,p3,p4,p5,p6
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [61, 81, 13, 291, 14, 178]         # map to p1..p6 for ratio

    NOSE_TIP = 1
    CHIN = 152
    DEFAULT_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )

    def __init__(self, config: Optional[RuntimeConfig] = None, model_path: str = "models/face_landmarker.task"):
        self.cfg = config or RuntimeConfig()
        self.model_path = Path(model_path)
        self._ensure_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self.state = DetectionState()
        self._eyes_closed_running = False
        self._head_down_running = False
        self._blink_running = False
        self._blink_start_t = 0.0
        self._last_yawn_t = 0.0
        self._fps_smooth = 20.0

    def _ensure_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_path.exists():
            return
        urlretrieve(self.DEFAULT_MODEL_URL, self.model_path)

    def _to_np(self, landmarks, width: int, height: int) -> np.ndarray:
        pts = np.zeros((len(landmarks), 2), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            pts[i] = [lm.x * width, lm.y * height]
        return pts

    def _extract(self, pts: np.ndarray, idxs: List[int]) -> List[np.ndarray]:
        return [pts[i] for i in idxs]

    def _calc_ear(self, pts: np.ndarray) -> float:
        l = self._extract(pts, self.LEFT_EYE)
        r = self._extract(pts, self.RIGHT_EYE)
        return 0.5 * (ratio_from_pairs(*l) + ratio_from_pairs(*r))

    def _calc_mar(self, pts: np.ndarray) -> float:
        m = self._extract(pts, self.MOUTH)
        return ratio_from_pairs(*m)

    def _pitch_deg(self, pts: np.ndarray) -> float:
        nose = pts[self.NOSE_TIP]
        chin = pts[self.CHIN]
        vec = chin - nose
        # Positive pitch when chin is lower than nose, larger means downward tilt.
        return float(np.degrees(np.arctan2(vec[1], abs(vec[0]) + 1e-6)) - 90.0)

    def _update_temporal_events(self, ear: float, mar: float, pitch_deg: float, dt: float, now: float) -> None:
        s = self.state
        s.event = ""

        # Eye closure and blink logic
        if ear < self.cfg.ear_closed_threshold:
            s.eyes_closed_s += dt
            self._eyes_closed_running = True
            if not self._blink_running:
                self._blink_running = True
                self._blink_start_t = now
        else:
            if self._blink_running:
                blink_dur = now - self._blink_start_t
                if self.cfg.blink_min_duration_s <= blink_dur <= self.cfg.blink_max_duration_s:
                    s.blink_count += 1
            self._blink_running = False
            s.eyes_closed_s = 0.0
            self._eyes_closed_running = False

        # Yawn logic (debounced to avoid counting same yawn repeatedly)
        if mar > self.cfg.mar_yawn_threshold and (now - self._last_yawn_t) > 1.5:
            s.yawn_count += 1
            self._last_yawn_t = now
            s.event = "YAWN"

        # Head down logic
        if pitch_deg > self.cfg.head_pitch_threshold_deg:
            s.head_down_s += dt
            self._head_down_running = True
        else:
            s.head_down_s = 0.0
            self._head_down_running = False

    def _compute_risk_and_status(self) -> None:
        s = self.state

        risk = 0.0
        status_tokens = []

        if s.eyes_closed_s >= self.cfg.eyes_closed_seconds_alert:
            risk += 50
            status_tokens.append("DROWSY")
            if not s.event:
                s.event = "EYES_CLOSED"

        if s.mar > self.cfg.mar_yawn_threshold:
            risk += 25
            status_tokens.append("YAWNING")
            if not s.event:
                s.event = "YAWNING"

        if s.head_down_s >= self.cfg.distracted_seconds_alert:
            risk += 25
            status_tokens.append("DISTRACTED")
            if not s.event:
                s.event = "HEAD_DOWN"

        # Mild risk from frequent blinks and yawns (rolling signal approximation)
        risk += min(20.0, s.yawn_count * 1.5)
        risk += min(15.0, s.blink_count * 0.15)

        s.risk_score = float(min(100.0, risk))
        s.is_alert = s.risk_score >= self.cfg.risk_alert_threshold
        if s.is_alert:
            s.status = "ALERT"
        elif status_tokens:
            s.status = "|".join(sorted(set(status_tokens)))
        else:
            s.status = "ATTENTIVE"

    def process_frame(self, frame_bgr: np.ndarray, dt: float, now: float) -> Tuple[np.ndarray, DetectionState]:
        s = self.state
        s.event = ""

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.face_mesh.detect_for_video(mp_image, int(now * 1000))

        if not result.face_landmarks:
            s.status = "NO_FACE"
            s.risk_score = max(0.0, s.risk_score - 2.0)
            s.is_alert = False
            return frame_bgr, s

        face_landmarks = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        pts = self._to_np(face_landmarks, w, h)

        s.ear = self._calc_ear(pts)
        s.mar = self._calc_mar(pts)
        s.pitch_deg = self._pitch_deg(pts)

        self._update_temporal_events(s.ear, s.mar, s.pitch_deg, dt, now)
        self._compute_risk_and_status()

        # Draw a lightweight subset of landmarks for visualization.
        draw_idxs = set(self.LEFT_EYE + self.RIGHT_EYE + self.MOUTH + [self.NOSE_TIP, self.CHIN])
        for idx in draw_idxs:
            p = pts[idx].astype(int)
            cv2.circle(frame_bgr, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

        return frame_bgr, s

    def close(self) -> None:
        self.face_mesh.close()

