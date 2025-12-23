from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    bbox: tuple  # (x1, y1, x2, y2)
    embedding: np.ndarray | None = None
    jersey: int | None = None
    player_id: str | None = None
