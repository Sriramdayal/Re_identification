from sklearn.metrics.pairwise import cosine_similarity

class HybridReID:
    def __init__(self, threshold=0.75):
        self.gallery = {}   # id -> embedding
        self.counter = 0
        self.threshold = threshold

    def assign(self, embedding, jersey=None):
        # Rule 1: OCR wins
        if jersey is not None:
            pid = f"jersey_{jersey}"
            self.gallery[pid] = embedding
            return pid

        # Rule 2: appearance match
        best_id, best_score = None, 0
        for pid, feat in self.gallery.items():
            score = cosine_similarity(
                [embedding], [feat]
            )[0][0]
            if score > best_score:
                best_id, best_score = pid, score

        if best_score > self.threshold:
            print(f"reidentified: {best_id}")
            return best_id

        # Rule 3: new identity
        pid = f"temp_{self.counter}"
        self.gallery[pid] = embedding
        self.counter += 1
        return pid
