import time
import config

class FatigueAnalyzer:
    def __init__(self):
        self.alert_timestamps = []

    def add_alert(self):
        current_time = time.time()
        self.alert_timestamps.append(current_time)

    def is_fatigue_critical(self):
        current_time = time.time()
        
        self.alert_timestamps = [t for t in self.alert_timestamps if current_time - t <= config.FATIGUE_WINDOW_SEC]

        return len(self.alert_timestamps) >= config.FATIGUE_ALERT_LIMIT
    