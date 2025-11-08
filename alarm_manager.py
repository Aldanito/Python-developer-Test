import time
from config import ALARM_DELAY_SECONDS


class AlarmManager:
    def __init__(self):
        self.is_alarm_active = False
        self.active_tracks = set()
        self.alarm_end_time = None
    
    def update(self, intruding_track_ids):
        intruding_track_ids = set(intruding_track_ids)
        current_time = time.time()
        
        self.active_tracks = intruding_track_ids
        
        if self.active_tracks:
            # someone is in a zone, activate alarm
            self.is_alarm_active = True
            self.alarm_end_time = None
        else:
            # no one in zone, but keep alarm on for delay period
            if self.is_alarm_active:
                if self.alarm_end_time is None:
                    self.alarm_end_time = current_time + ALARM_DELAY_SECONDS
                elif current_time >= self.alarm_end_time:
                    self.is_alarm_active = False
                    self.alarm_end_time = None
    
    def get_alarm_status(self):
        return self.is_alarm_active

