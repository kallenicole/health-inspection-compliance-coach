# api/services/model_service.py
import os, json
from typing import Dict, Any
import pandas as pd

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

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

        # Load seed + features + ML model
        self._demo = self._load_demo_seed()
        self.rat_features = self._load_rat_features()
        self.ml_model = self._load_ml_model()

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
        def _f(r, col):
            v = getattr(r, col, None)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _i(r, col):
            v = getattr(r, col, None)
            try:
                return int(v) if v is not None else 0
            except (TypeError, ValueError):
                return 0

        for r in df.itertuples(index=False):
            out[r.camis] = {
                "rat_index":              _f(r, "rat_index"),
                "pest_index":             _f(r, "pest_index"),
                "rat311_cnt_180d_k1":     _i(r, "rat311_cnt_180d_k1"),
                "mouse311_cnt_180d_k1":   _i(r, "mouse311_cnt_180d_k1"),
                "pest311_cnt_180d_k1":    _i(r, "pest311_cnt_180d_k1"),
                "ratinsp_fail_365d_k1":   _i(r, "ratinsp_fail_365d_k1"),
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
        except Exception as e:
            print(f"[model_service] WARNING: failed to load rat features: {e}", flush=True)
        return {}

    def _load_ml_model(self):
        """Load trained scikit-learn model from MODEL_PATH; returns None if unavailable."""
        if not _JOBLIB_AVAILABLE:
            return None
        try:
            path = self.model_path
            if path and os.path.exists(path):
                model = joblib.load(path)
                print(f"[model_service] ML model loaded from {path}", flush=True)
                return model
        except Exception as e:
            print(f"[model_service] Could not load ML model: {e}", flush=True)
        return None

    def ml_predict(self, features_df) -> float:
        """Return predicted prob_bc for a single-row features DataFrame. Raises if model unavailable."""
        if self.ml_model is None:
            raise RuntimeError("ML model not loaded")
        return float(self.ml_model.predict_proba(features_df)[0, 1])

    def reload_rat_features(self) -> int:
        """Reload features into memory; return count."""
        self.rat_features = self._load_rat_features()
        return len(self.rat_features)

    # ---------- heuristics ----------

    def _apply_rat_heuristics(self, camis: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        rf = self.rat_features.get(camis, {})
        payload.setdefault("rat_index",            rf.get("rat_index"))
        payload.setdefault("pest_index",           rf.get("pest_index"))
        payload.setdefault("rat311_cnt_180d_k1",   rf.get("rat311_cnt_180d_k1"))
        payload.setdefault("mouse311_cnt_180d_k1", rf.get("mouse311_cnt_180d_k1"))
        payload.setdefault("pest311_cnt_180d_k1",  rf.get("pest311_cnt_180d_k1"))
        payload.setdefault("ratinsp_fail_365d_k1", rf.get("ratinsp_fail_365d_k1"))

        # Use pest_index (broader signal) for risk bump; fall back to rat_index
        pi = rf.get("pest_index") or rf.get("rat_index")
        if isinstance(pi, (int, float)):
            if isinstance(payload.get("prob_bc"), (int, float)):
                payload["prob_bc"] = float(min(0.99, payload["prob_bc"] + min(0.12, 0.12 * pi)))
            tvp = payload.get("top_violation_probs") or []
            for v in tvp:
                if v.get("code") == "04L" and isinstance(v.get("probability"), (int, float)):
                    v["probability"] = float(min(0.99, v["probability"] + 0.20 * pi))
        return payload

    # ---------- public ----------

    def score_camis(self, camis: str) -> Dict[str, Any]:
        camis = str(camis)
        if camis not in self._demo:
            raise KeyError("CAMIS not available (seed only in MVP)")

        payload = dict(self._demo[camis])  # copy
        payload = self._apply_rat_heuristics(camis, payload)
        return payload
