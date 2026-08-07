from ultralytics import YOLO
import config

class DrowsinessDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)

    def process_frame(self, frame):
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)

        current_status = "awake"
        max_conf = 0.0

        for result in results:
            for box in result.boxes: # type: ignore
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                conf = float(box.conf[0])

                if label == "drowsy" and conf > max_conf:
                    current_status = "drowsy"
                    max_conf = conf

        return current_status, max_conf

    