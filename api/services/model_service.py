import os, json
from typing import Dict, Any

class ModelService:
    def __init__(self, model_path: str, demo_seed: str):
        self.model_path = model_path
        self.demo_seed = demo_seed
        # Fallback baked seed inside the image for first run
        self.baked_seed = os.getenv("BAKED_DEMO_SEED_FILE", "./data/demo_seed.json")
        self._demo = self._load_demo_seed()

    def _load_demo_seed(self):
        # Prefer runtime-refreshed seed (e.g., /tmp/data/demo_seed.json in Cloud Run)
        if self.demo_seed and os.path.exists(self.demo_seed):
            with open(self.demo_seed, "r") as f:
                return json.load(f)
        # Fallback to baked seed shipped in the image
        if self.baked_seed and os.path.exists(self.baked_seed):
            with open(self.baked_seed, "r") as f:
                return json.load(f)
        return {}

    def score_camis(self, camis: str) -> Dict[str, Any]:
        if camis in self._demo:
            return self._demo[camis]
        raise KeyError("CAMIS not available (seed only in MVP)")
