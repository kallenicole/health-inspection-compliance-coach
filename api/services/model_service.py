# api/services/model_service.py
import os, json
from typing import Dict, Any, Optional
import pandas as pd

class ModelService:
    def __init__(self, model_path: str, demo_seed: str):
        self.model_path = model_path
        self.demo_seed = demo_seed

        # Directories
        self.feature_dir = os.getenv("FEATURE_STORE_DIR", "./data/parquet")      # runtime (e.g., /tmp)
        self.baked_dir = os.getenv("BAKED_FEATURE_DIR", "/app/data/parquet")     # image-baked fallback

        # Seeds
        self.baked_seed = os.getenv("BAKED_DEMO_SEED_FILE", "./data/demo_seed.json")
        self._demo = self._load_demo_seed()

        # Rat features in memory
        self.rat_features: Dict[str, Dict[str, Any]] = {}
        self.rat_source: str = ""
        self.reload_rat_features()

    # --------- seeds ----------
    def _load_demo_seed(self) -> Dict[str, Any]:
        if self.demo_seed and os.path.exists(self.demo_seed):
            with open(self.demo_seed, "r") as f:
                return json.load(f)
        if self.baked_seed and os.path.exists(self.baked_seed):
            with open(self.baked_seed, "r") as f:
                return json.load(f)
        return {}

    # --------- rat features ----------
    def _rat_parquet_candidates(self):
        return [
            os.path.join(self.feature_dir, "rat_index.parquet"),  # prefer fresh runtime
            os.path.join(self.baked_dir, "rat_index.parquet"),    # fallback baked
        ]

    def _load_rat_features_from(self, path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        if df.empty:
            return {}
        df["camis"] = df["camis"].astype(str)
        out: Dict[str, Dict[str, Any]] = {}
        for r in df.itertuples(index=False):
            out[r.camis] = {
                "rat_index": float(getattr(r, "rat_index", None)) if getattr(r, "rat_index", None) is not None else None,
                "rat311_cnt_180d_k1": int(getattr(r, "rat311_cnt_180d_k1", 0) or 0),
                "ratinsp_fail_365d_k1": int(getattr(r, "ratinsp_fail_365d_k1", 0) or 0),
            }
        return out

    def reload_rat_features(self) -> int:
        """
        Try runtime parquet, then baked parquet. Returns number of rows loaded.
        """
        for p in self._rat_parquet_candidates():
            try:
                data = self._load_rat_features_from(p)
                if data is not None:
                    self.rat_features = data
                    self.rat_source = p
                    return len(self.rat_features)
            except Exception:
                # swallow and try next candidate
                pass
        self.rat_features = {}
        self.rat_source = ""
        return 0

    def _apply_rat_heuristics(self, camis: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        rf = self.rat_features.get(str(camis))
        if rf:
            # attach raw features
            payload.setdefault("rat_index", rf.get("rat_index"))
            payload.setdefault("rat311_cnt_180d_k1", rf.get("rat311_cnt_180d_k1"))
            payload.setdefault("ratinsp_fail_365d_k1", rf.get("ratinsp_fail_365d_k1"))

            # optional tiny heuristic bump to prob_bc and 04L
            ri = rf.get("rat_index")
            if isinstance(ri, (int, float)):
                if "prob_bc" in payload and isinstance(payload["prob_bc"], (int, float)):
                    payload["prob_bc"] = float(min(0.99, payload["prob_bc"] + min(0.12, 0.12 * ri)))
                tvp = payload.get("top_violation_probs") or []
                for v in tvp:
                    if v.get("code") == "04L" and isinstance(v.get("probability"), (int, float)):
                        v["probability"] = float(min(0.99, v["probability"] + 0.20 * ri))
        else:
            # ensure fields exist even if we have no features
            payload.setdefault("rat_index", None)
            payload.setdefault("rat311_cnt_180d_k1", None)
            payload.setdefault("ratinsp_fail_365d_k1", None)
        return payload

    # --------- main scoring ----------
    def score_camis(self, camis: str) -> Dict[str, Any]:
        camis = str(camis)
        if camis not in self._demo:
            raise KeyError("CAMIS not available (seed only in MVP)")
        payload = dict(self._demo[camis])  # copy
        payload = self._apply_rat_heuristics(camis, payload)
        return payload
