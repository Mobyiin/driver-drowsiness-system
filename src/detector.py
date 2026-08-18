import math
from ultralytics import YOLO
import torch
import gc
import config

class DrowsinessDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        frame_center_x = w / 2.0
        frame_center_y = h / 2.0
        max_dist = math.hypot(frame_center_x, frame_center_y)

        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)

        best_detection = None
        highest_score = -1.0

        for result in results:
            for box in result.boxes:  # type: ignore
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                area = (x2 - x1) * (y2 - y1)
                
                dist_to_center = math.hypot(center_x - frame_center_x, center_y - frame_center_y)
            

                normalized_dist = dist_to_center / max_dist
                

                score = area * (1.0 - normalized_dist)
                
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                conf = float(box.conf[0])

                if score > highest_score:
                    highest_score = score
                    best_detection = {
                        "status": label,
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    }

        if best_detection is None:
            return "no_target", 0.0, None

        return best_detection["status"], best_detection["confidence"], best_detection["bbox"]