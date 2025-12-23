from sklearn.metrics.pairwise import cosine_similarity

class ReIDMatcher:
    def __init__(self, threshold: float):
        self.gallery = {}
        self.threshold = threshold
        self.counter = 0

    def assign(self, embedding, jersey=None):
        if jersey is not None:
            pid = f"jersey_{jersey}"
            self.gallery[pid] = embedding
            return pid

        best_id, best_score = None, 0
        for pid, feat in self.gallery.items():
            score = cosine_similarity([embedding], [feat])[0][0]
            if score > best_score:
                best_id, best_score = pid, score

        if best_score > self.threshold:
            return best_id

        pid = f"temp_{self.counter}"
        self.gallery[pid] = embedding
        self.counter += 1
        return pid
