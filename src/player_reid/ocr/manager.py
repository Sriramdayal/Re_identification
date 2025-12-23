from collections import Counter, deque

class OCRManager:
    def __init__(self, history_len=10, min_conf_count=3):
        """
        Manages rolling history of OCR results per track_id.
        """
        self.history = {} # track_id -> deque([jersey_num, ...])
        self.history_len = history_len
        self.min_conf_count = min_conf_count

    def update(self, track_id, jersey_num):
        """
        Add a new observation for a track.
        """
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.history_len)
        
        # Only add valid numbers
        if jersey_num is not None:
             self.history[track_id].append(jersey_num)

    def get_stable_jersey(self, track_id):
        """
        Returns the majority voted jersey number if confidence is high enough.
        """
        if track_id not in self.history or len(self.history[track_id]) == 0:
            return None
            
        counts = Counter(self.history[track_id])
        most_common, count = counts.most_common(1)[0]
        
        # Threshold: Need at least N consistent reads
        if count >= self.min_conf_count:
            return most_common
            
        return None
