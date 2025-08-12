# api/services/model_service.py
import os, json
from typing import Dict, Any, Optional
import pandas as pd

class ModelService:
    def __init__(self, model_path: str, demo_seed: str):
        self.model_path = model_path
        self.demo_seed = demo_seed

        # baked seed fallback (inside the container)
        self.baked_seed = os.getenv("BAKED_DEMO_SEED_FILE", "./data/demo_seed.json")
        self.feature_dir = os.getenv("FEATURE_STORE_DIR", "./data/parquet")
        self._demo = self._load_demo_seed()

        # Load rat features into memory (optional if file missing)
        self.rat_features = self._load_rat_features()

    def _load_demo_seed(self):
        if self.demo_seed and os.path.exists(self.demo_seed):
            with open(self.demo_seed, "r") as f:
                return json.load(f)
        if self.baked_seed and os.path.exists(self.baked_seed):
            with open(self.baked_seed, "r") as f:
                return json.load(f)
        return {}

    def _load_rat_features(self) -> Dict[str, Dict[str, Any]]:
        try:
            path = os.path.join(self.feature_dir, "rat_index.parquet")
            if not os.path.exists(path):
                return {}
            df = pd.read_parquet(path)
            df["camis"] = df["camis"].astype(str)
            return {
                r.camis: {
                    "rat_index": float(r.rat_index) if r.rat_index is not None else None,
                    "rat311_cnt_180d_k1": int(r.rat311_cnt_180d_k1) if r.rat311_cnt_180d_k1 is not None else 0,
                    "ratinsp_fail_365d_k1": int(r.ratinsp_fail_365d_k1) if r.ratinsp_fail_365d_k1 is not None else 0,
                }
                for r in df.itertuples(index=False)
            }
        except Exception:
            # Keep service alive if parquet not present or unreadable
            return {}

    def _apply_rat_heuristics(self, camis: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        rf = self.rat_features.get(camis, {})
        if not rf:
            return payload

        # attach raw features
        payload.update({
            "rat_index": rf.get("rat_index"),
            "rat311_cnt_180d_k1": rf.get("rat311_cnt_180d_k1"),
            "ratinsp_fail_365d_k1": rf.get("ratinsp_fail_365d_k1"),
        })

        # Heuristic bumps (temporary until a trained model uses rat_index as a feature)
        ri = rf.get("rat_index")
        if ri is None:
            return payload

        # bump overall B/C probability slightly (cap +0.12)
        if "prob_bc" in payload and isinstance(payload["prob_bc"], (int, float)):
            payload["prob_bc"] = float(min(0.99, payload["prob_bc"] + min(0.12, 0.12 * ri)))

        # bump mice (04L) violation probability slightly (cap to 0.99)
        tvp = payload.get("top_violation_probs") or []
        for v in tvp:
            if v.get("code") == "04L" and isinstance(v.get("probability"), (int, float)):
                v["probability"] = float(min(0.99, v["probability"] + 0.20 * ri))  # up to +20 points

        return payload

    def score_camis(self, camis: str) -> Dict[str, Any]:
        camis = str(camis)
        if camis not in self._demo:
            raise KeyError("CAMIS not available (seed only in MVP)")

        payload = dict(self._demo[camis])  # copy
        payload = self._apply_rat_heuristics(camis, payload)
        return payload
