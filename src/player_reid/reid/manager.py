from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ReIDManager:
    def __init__(self, threshold=0.75):
        self.gallery = {}   # pid -> embedding (centroid)
        self.track_map = {} # track_id -> pid
        self.track_embeddings = {} # track_id -> current_embedding (EMA)
        
        self.counter = 0
        self.threshold = threshold
        self.alpha = 0.9 # EMA factor

    def update_track_embedding(self, track_id, embedding):
        """
        Update the running EMA embedding for a track.
        """
        if track_id not in self.track_embeddings:
            self.track_embeddings[track_id] = embedding
        else:
            # EMA Update
            prev = self.track_embeddings[track_id]
            self.track_embeddings[track_id] = self.alpha * prev + (1 - self.alpha) * embedding

    def resolve_identity(self, track_id, jersey_num=None):
        """
        Decide the identity for a track.
        If track already has an ID, return it (Persistence).
        If not, try to match via Jersey -> Appearance -> New ID.
        """
        # 1. Persistence (Stability)
        if track_id in self.track_map:
            # Optionally verify if jersey matches, but usually we trust the track
            if jersey_num is not None:
                # Conflict check could go here
                pass
            return self.track_map[track_id]
            
        current_feat = self.track_embeddings.get(track_id)
        if current_feat is None:
            return "unknown" 

        # 2. Jersey Match (Strong Signal)
        if jersey_num is not None:
            pid = f"jersey_{jersey_num}"
            self._register_identity(track_id, pid, current_feat)
            return pid

        # 3. Appearance Match (Weak Signal)
        best_pid, best_score = None, 0
        for pid, center_feat in self.gallery.items():
            score = cosine_similarity([current_feat], [center_feat])[0][0]
            if score > best_score:
                best_pid, best_score = pid, score
        
        if best_score > self.threshold:
            print(f"[ReID] Matched track {track_id} to {best_pid} ({best_score:.2f})")
            self._register_identity(track_id, best_pid, current_feat)
            return best_pid

        # 4. New Identity
        new_pid = f"temp_{self.counter}"
        self.counter += 1
        self._register_identity(track_id, new_pid, current_feat)
        return new_pid

    def _register_identity(self, track_id, pid, embedding):
        self.track_map[track_id] = pid
        # In a real system, we might average features for the gallery
        # Here we just set it if it's new
        if pid not in self.gallery:
            self.gallery[pid] = embedding
