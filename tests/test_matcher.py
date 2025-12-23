import numpy as np
from player_reid.reid.matcher import ReIDMatcher

def test_assignment():
    m = ReIDMatcher(0.8)
    emb = np.random.rand(512)
    pid = m.assign(emb)
    assert pid.startswith("temp_")
