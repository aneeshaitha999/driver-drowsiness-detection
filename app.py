

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2

from detector import DrowsinessDetector
from utils import CSVEventLogger, RuntimeConfig, draw_status_panel, ensure_dir, format_fps, overlay_warning


def play_alarm_async(alarm_path: Path) -> None:
    """Play alarm asynchronously; fallback to terminal bell if file unavailable."""
    try:
        import pygame

        if alarm_path.exists():
            pygame.mixer.init()
            pygame.mixer.music.load(str(alarm_path))
            pygame.mixer.music.play()
            return
    except Exception:
        pass

    # Basic fallback: terminal bell
    print("\a", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI-Powered Driver Drowsiness Detection System")
    p.add_argument("--camera-index", type=int, default=0, help="Webcam index to use")
    p.add_argument("--width", type=int, default=960, help="Capture width")
    p.add_argument("--height", type=int, default=540, help="Capture height")
    p.add_argument("--mirror", action="store_true", help="Mirror preview feed")
    p.add_argument("--show-fps", action="store_true", help="Render FPS counter")
    p.add_argument("--log-every-n-frames", type=int, default=5, help="CSV logging interval")
    p.add_argument("--alarm", type=str, default="alarm.wav", help="Path to alarm wav file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir("logs")

    cfg = RuntimeConfig()
    detector = DrowsinessDetector(cfg)
    logger = CSVEventLogger(logs_dir="logs")

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check camera permissions and index.")

    print(f"Logging metrics to: {logger.filepath}")
    print("Controls: [q] quit | [r] reset counters")

    prev_ts = time.time()
    frame_idx = 0
    alarm_latched = False
    alarm_path = Path(args.alarm)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera; retrying.")
                time.sleep(0.05)
                continue

            now = time.time()
            dt = max(0.001, now - prev_ts)
            prev_ts = now
            fps = format_fps(now - dt, now)

            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame, state = detector.process_frame(frame, dt=dt, now=now)

            lines = [
                (f"EAR: {state.ear:.3f}", (20, 220, 20)),
                (f"MAR: {state.mar:.3f}", (20, 220, 20)),
                (f"Pitch: {state.pitch_deg:.1f} deg", (20, 220, 20)),
                (f"Risk: {state.risk_score:.1f}/100", (0, 165, 255) if state.risk_score > 50 else (20, 220, 20)),
                (f"Status: {state.status}", (0, 0, 255) if state.is_alert else (255, 255, 0)),
                (f"Blinks: {state.blink_count} | Yawns: {state.yawn_count}", (255, 255, 255)),
            ]
            if args.show_fps:
                lines.append((f"FPS: {fps:.1f}", (255, 255, 255)))
            draw_status_panel(frame, lines)

            if state.is_alert:
                overlay_warning(frame)
                if not alarm_latched:
                    threading.Thread(target=play_alarm_async, args=(alarm_path,), daemon=True).start()
                    alarm_latched = True
            else:
                alarm_latched = False

            if frame_idx % max(1, args.log_every_n_frames) == 0:
                logger.log(
                    ear=state.ear,
                    mar=state.mar,
                    pitch_deg=state.pitch_deg,
                    risk_score=state.risk_score,
                    blink_count=state.blink_count,
                    yawn_count=state.yawn_count,
                    status=state.status,
                    event=state.event,
                )
            frame_idx += 1

            cv2.imshow("AI Driver Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                detector.state.blink_count = 0
                detector.state.yawn_count = 0
                detector.state.risk_score = 0.0
                print("Counters reset.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()

