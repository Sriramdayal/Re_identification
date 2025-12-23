import torch
from torchreid.utils import FeatureExtractor

class AppearanceEmbedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            device=device
        )

    def embed(self, crops):
        feats = self.extractor(crops)
        return feats.cpu().numpy()
