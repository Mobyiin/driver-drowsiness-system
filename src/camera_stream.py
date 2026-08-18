import cv2
import threading
import time

class CameraStream:
    def __init__(self, source=0, target_size=(640, 640)):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.target_size = target_size
        
        self.grabbed, frame = self.cap.read()
        self.frame = cv2.resize(frame, self.target_size) if self.grabbed else None
        
        self.started = False
        self.read_lock = threading.Lock()

        self.fps_counter = 0
        self.capture_fps = 0.0
        self._fps_timer = time.time()
        
        self.frame_delay = 0.0 

    def set_fps_limit(self, max_fps):
        if max_fps > 0:
            self.frame_delay = 1.0 / max_fps
        else:
            self.frame_delay = 0.0 

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.started:
            start_loop_time = time.time()
            
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.started = False
                break
            
            resized_frame = cv2.resize(frame, self.target_size)
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = resized_frame

            self.fps_counter += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self.capture_fps = self.fps_counter / (now - self._fps_timer)
                self.fps_counter = 0
                self._fps_timer = now

            elapsed = time.time() - start_loop_time
            if self.frame_delay > elapsed:
                time.sleep(self.frame_delay - elapsed)

    def read(self):
        with self.read_lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()