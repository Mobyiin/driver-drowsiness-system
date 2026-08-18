import cv2
import time
import torch
import config
from src.camera_stream import CameraStream
from src.detector import DrowsinessDetector
from src.activeidle_sm import ActiveIdleStateMachine
from src.fatigue_analyzer import FatigueAnalyzer
from src.audio_player import AudioPlayer
from utils.helpers import render_ui

torch.set_num_threads(2)

def main():
    detector = DrowsinessDetector()
    fatigue_analyzer = FatigueAnalyzer()
    audio = AudioPlayer() 

    ActiveIdle = ActiveIdleStateMachine(idle_timeout=10.0, active_fps=10.0, idle_fps=0.5)

    camera = CameraStream(source=config.CAMERA_INDEX, target_size=(640, 640)).start()
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    drowsy_start_time = None
    is_drowsy_active = False
    alert_triggered = False 

    last_inference_time = 0.0
    inference_counter = 0
    inference_fps = 0.0
    fps_timer = time.time()

    status = "no_target"
    conf = 0.0
    target_bbox = None
    previous_state = ActiveIdle.state

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                break

            current_time = time.time()
            current_state = ActiveIdle.state
            if current_state != previous_state:
                if current_state == "IDLE":
                    camera.set_fps_limit(config.IDLE_CAMERA_FPS) 
                else:
                    camera.set_fps_limit(0)
                previous_state = current_state


            current_interval = ActiveIdle.get_inference_interval()

            if current_time - last_inference_time >= current_interval:
                last_inference_time = current_time
                status, conf, target_bbox = detector.process_frame(frame)
                ActiveIdle.update(status)
                inference_counter += 1

            if current_time - fps_timer >= 1.0:
                inference_fps = inference_counter / (current_time - fps_timer)
                inference_counter = 0
                fps_timer = current_time

            is_drowsy = (status == "drowsy")
            drowsy_duration = 0.0

            if is_drowsy:
                if not is_drowsy_active:
                    is_drowsy_active = True
                    drowsy_start_time = current_time
                    alert_triggered = False
                
                if drowsy_start_time is not None:
                    drowsy_duration = current_time - drowsy_start_time

                if drowsy_duration >= config.DROWSY_TIME_LIMIT:
                    if not alert_triggered:
                        alert_triggered = True
                        fatigue_analyzer.add_alert()
            else:
                if is_drowsy_active:
                    is_drowsy_active = False
                    drowsy_start_time = None
                    alert_triggered = False
                    audio.stop_warning_sound()

            is_fatigued = fatigue_analyzer.is_fatigue_critical()

            if is_fatigued:
                audio.play_critical_sound()
            elif alert_triggered:
                audio.play_warning_sound()
            else:
                if not is_fatigued:
                    audio.stop_critical_sound()

            render_ui(
                frame=frame,
                is_fatigued=is_fatigued,
                alert_triggered=alert_triggered,
                is_drowsy_active=is_drowsy_active,
                drowsy_duration=drowsy_duration,
                capture_fps=camera.capture_fps,
                inference_fps=inference_fps,
                target_bbox=target_bbox
            )

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            try:
                if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    finally:
        audio.stop_all()
        camera.stop()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
