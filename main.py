import cv2
import time
import config
from src.detector import DrowsinessDetector
from src.fatigue_analyzer import FatigueAnalyzer
from src.audio_player import AudioPlayer
from utils.helpers import render_ui

def main():
    detector = DrowsinessDetector()
    fatigue_analyzer = FatigueAnalyzer()
    audio = AudioPlayer() 

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    drowsy_start_time = None
    is_drowsy_active = False
    alert_triggered = False 

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        status, conf = detector.process_frame(frame)
        is_drowsy = (status == "drowsy")
        current_time = time.time()
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

        render_ui(frame, is_fatigued, alert_triggered, is_drowsy_active, drowsy_duration)
        cv2.imshow(config.WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
                break
        except cv2.error:
            break

    audio.stop_all()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()