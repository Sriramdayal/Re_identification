import torch
from torchreid.utils import FeatureExtractor

class AppearanceEmbedder:
    def __init__(self, model="osnet_x1_0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = FeatureExtractor(
            model_name=model,
            device=self.device
        )

    def embed(self, images):
        feats = self.extractor(images)
        return feats.cpu().numpy()
