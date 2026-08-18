import time

class ActiveIdleStateMachine:
    def __init__(self, idle_timeout=10.0, active_fps=10.0, idle_fps=0.5):
        self.state = "IDLE"  
        self.idle_timeout = idle_timeout
        
        self.active_interval = 1.0 / active_fps 
        self.idle_interval = 1.0 / idle_fps
        
        self.last_target_time = time.time()

    def update(self, status):
        current_time = time.time()

        if status in ["awake", "drowsy"]:
            self.last_target_time = current_time
            self.state = "ACTIVE"
        else:
            if current_time - self.last_target_time >= self.idle_timeout:
                self.state = "IDLE"

    def get_inference_interval(self):
        if self.state == "ACTIVE":
            return self.active_interval
        else:
            return self.idle_interval
