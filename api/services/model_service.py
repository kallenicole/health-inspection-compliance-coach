# api/services/model_service.py
import os, json
from typing import Dict, Any
import pandas as pd

class ModelService:
    """
    Loads demo seed and rat features.
    Prefers runtime FEATURE_STORE_DIR (e.g. /tmp) but will fall back
    to a baked copy in BAKED_FEATURE_DIR (e.g. /app/data/parquet) on cold start.
    """

    def __init__(self, model_path: str, demo_seed: str):
        self.model_path = model_path
        self.demo_seed = demo_seed

        # Directories
        self.feature_dir = os.getenv("FEATURE_STORE_DIR", "./data/parquet")
        self.baked_feature_dir = os.getenv("BAKED_FEATURE_DIR", "./data/parquet")

        # Demo seed baked fallback (inside image)
        self.baked_seed = os.getenv("BAKED_DEMO_SEED_FILE", "./data/demo_seed.json")

        # Load seed + features
        self._demo = self._load_demo_seed()
        self.rat_features = self._load_rat_features()

    # ---------- loading ----------

    def _load_demo_seed(self) -> Dict[str, Any]:
        """Prefer runtime demo_seed; else baked copy; else {}."""
        try:
            if self.demo_seed and os.path.exists(self.demo_seed):
                with open(self.demo_seed, "r") as f:
                    return json.load(f)
            if self.baked_seed and os.path.exists(self.baked_seed):
                with open(self.baked_seed, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _read_rat_parquet(self, path: str) -> Dict[str, Dict[str, Any]]:
        df = pd.read_parquet(path)
        if df.empty:
            return {}
        df["camis"] = df["camis"].astype(str)
        out: Dict[str, Dict[str, Any]] = {}
        for r in df.itertuples(index=False):
            out[r.camis] = {
                "rat_index": float(getattr(r, "rat_index", None)) if getattr(r, "rat_index", None) is not None else None,
                "rat311_cnt_180d_k1": int(getattr(r, "rat311_cnt_180d_k1", 0)) if getattr(r, "rat311_cnt_180d_k1", None) is not None else 0,
                "ratinsp_fail_365d_k1": int(getattr(r, "ratinsp_fail_365d_k1", 0)) if getattr(r, "ratinsp_fail_365d_k1", None) is not None else 0,
            }
        return out

    def _load_rat_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Try runtime copy first (FEATURE_STORE_DIR/rat_index.parquet).
        If missing, fall back to baked copy (BAKED_FEATURE_DIR/rat_index.parquet).
        """
        try:
            runtime_path = os.path.join(self.feature_dir, "rat_index.parquet")
            baked_path = os.path.join(self.baked_feature_dir, "rat_index.parquet")

            if os.path.exists(runtime_path):
                return self._read_rat_parquet(runtime_path)
            if os.path.exists(baked_path):
                return self._read_rat_parquet(baked_path)
        except Exception:
            pass
        return {}

    def reload_rat_features(self) -> int:
        """Reload features into memory; return count."""
        self.rat_features = self._load_rat_features()
        return len(self.rat_features)

    # ---------- heuristics ----------

    def _apply_rat_heuristics(self, camis: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        rf = self.rat_features.get(camis, {})
        # Always attach the raw feature fields (even if None/0) so UI renders consistently.
        payload.setdefault("rat_index", rf.get("rat_index"))
        payload.setdefault("rat311_cnt_180d_k1", rf.get("rat311_cnt_180d_k1"))
        payload.setdefault("ratinsp_fail_365d_k1", rf.get("ratinsp_fail_365d_k1"))

        ri = rf.get("rat_index")
        if isinstance(ri, (int, float)):
            # Gentle overall risk bump, capped
            if isinstance(payload.get("prob_bc"), (int, float)):
                payload["prob_bc"] = float(min(0.99, payload["prob_bc"] + min(0.12, 0.12 * ri)))
            # Slightly bump mice (04L) probability if present
            tvp = payload.get("top_violation_probs") or []
            for v in tvp:
                if v.get("code") == "04L" and isinstance(v.get("probability"), (int, float)):
                    v["probability"] = float(min(0.99, v["probability"] + 0.20 * ri))
        return payload

    # ---------- public ----------

    def score_camis(self, camis: str) -> Dict[str, Any]:
        camis = str(camis)
        if camis not in self._demo:
            raise KeyError("CAMIS not available (seed only in MVP)")

        payload = dict(self._demo[camis])  # copy
        payload = self._apply_rat_heuristics(camis, payload)
        return payload
