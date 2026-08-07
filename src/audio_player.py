import pygame
import os
import time
import config

class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.warning_sound = self._load_sound(config.WARNING_SOUND_PATH)
        self.critical_sound = self._load_sound(config.CRITICAL_SOUND_PATH)

        self.channel_warning = pygame.mixer.Channel(0)
        self.channel_critical = pygame.mixer.Channel(1)

        self.critical_start_time = None

    def _load_sound(self,sound_path):
        if os.path.exists(sound_path):
            return pygame.mixer.Sound(sound_path)
        else:
            print(f"Sound file not found: {sound_path}")
            return None

    def play_warning_sound(self):
       if self.warning_sound and not self.channel_warning.get_busy():
            self.channel_warning.play(self.warning_sound, loops=-1)

    def stop_warning_sound(self):
        if self.warning_sound:
            self.channel_warning.stop()

    def play_critical_sound(self):
        current_time = time.time()
        
        if not self.channel_critical.get_busy() and self.critical_start_time is None:
            self.stop_warning_sound()
            if self.critical_sound:
                self.channel_critical.play(self.critical_sound)
                self.critical_start_time = current_time

        
        if self.critical_start_time is not None:
            if current_time - self.critical_start_time >= config.CRITICAL_SOUND_DURATION_SEC :
                self.stop_critical_sound()

    def stop_critical_sound(self):
        if self.critical_sound:
            self.channel_critical.stop()
        self.critical_start_time = None

    def stop_all(self):
        self.stop_warning_sound()
        self.stop_critical_sound()
    