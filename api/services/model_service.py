# api/services/model_service.py
import os, json
from typing import Dict, Any
import pandas as pd


class ModelService:
    """
    Loads demo seed, rat features, and pre-computed ML scores.

    ML scores are produced by etl/precompute_scores.py (runs after nightly ETL)
    and stored in ml_scores.parquet — a simple {camis → prob_bc} lookup.
    This avoids loading scikit-learn at runtime, saving ~80 MB of startup RSS.

    Prefers runtime FEATURE_STORE_DIR (e.g. /tmp on Cloud Run / Railway)
    but falls back to baked copies in BAKED_FEATURE_DIR on cold start.
    """

    def __init__(self, model_path: str, demo_seed: str):
        self.model_path = model_path  # kept for reference / admin tooling
        self.demo_seed  = demo_seed

        self.feature_dir       = os.getenv("FEATURE_STORE_DIR",   "./data/parquet")
        self.baked_feature_dir = os.getenv("BAKED_FEATURE_DIR",   "./data/parquet")
        self.baked_seed        = os.getenv("BAKED_DEMO_SEED_FILE", "./data/demo_seed.json")

        self._demo        = self._load_demo_seed()
        self.rat_features = self._load_rat_features()
        self.ml_scores    = self._load_ml_scores()

    # ---------- loading ----------

    def _load_demo_seed(self) -> Dict[str, Any]:
        """Prefer runtime demo_seed; else baked copy; else {}."""
        try:
            for path in (self.demo_seed, self.baked_seed):
                if path and os.path.exists(path):
                    with open(path) as f:
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
            try:   return float(v) if v is not None else None
            except: return None

        def _i(r, col):
            v = getattr(r, col, None)
            try:   return int(v) if v is not None else 0
            except: return 0

        for r in df.itertuples(index=False):
            out[r.camis] = {
                "rat_index":            _f(r, "rat_index"),
                "pest_index":           _f(r, "pest_index"),
                "rat311_cnt_180d_k1":   _i(r, "rat311_cnt_180d_k1"),
                "mouse311_cnt_180d_k1": _i(r, "mouse311_cnt_180d_k1"),
                "pest311_cnt_180d_k1":  _i(r, "pest311_cnt_180d_k1"),
                "ratinsp_fail_365d_k1": _i(r, "ratinsp_fail_365d_k1"),
            }
        return out

    def _load_rat_features(self) -> Dict[str, Dict[str, Any]]:
        try:
            for path in (
                os.path.join(self.feature_dir,       "rat_index.parquet"),
                os.path.join(self.baked_feature_dir, "rat_index.parquet"),
            ):
                if os.path.exists(path):
                    return self._read_rat_parquet(path)
        except Exception as e:
            print(f"[model_service] WARNING: failed to load rat features: {e}", flush=True)
        return {}

    def _load_ml_scores(self) -> Dict[str, float]:
        """
        Load pre-computed prob_bc scores from ml_scores.parquet.
        Returns {} if file not found (heuristic will be used as fallback).
        """
        try:
            for path in (
                os.path.join(self.feature_dir,       "ml_scores.parquet"),
                os.path.join(self.baked_feature_dir, "ml_scores.parquet"),
            ):
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    df["camis"] = df["camis"].astype(str)
                    scores = dict(zip(df["camis"], df["prob_bc"].astype(float)))
                    print(f"[model_service] ML scores loaded: {len(scores):,} restaurants from {path}", flush=True)
                    return scores
        except Exception as e:
            print(f"[model_service] WARNING: failed to load ML scores: {e}", flush=True)
        return {}

    # ---------- reloading ----------

    def reload_rat_features(self) -> int:
        self.rat_features = self._load_rat_features()
        return len(self.rat_features)

    def reload_ml_scores(self) -> int:
        self.ml_scores = self._load_ml_scores()
        return len(self.ml_scores)

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
            for v in (payload.get("top_violation_probs") or []):
                if v.get("code") == "04L" and isinstance(v.get("probability"), (int, float)):
                    v["probability"] = float(min(0.99, v["probability"] + 0.20 * pi))
        return payload

    # ---------- public ----------

    def score_camis(self, camis: str) -> Dict[str, Any]:
        camis = str(camis)
        if camis not in self._demo:
            raise KeyError("CAMIS not in demo seed")
        payload = dict(self._demo[camis])
        payload = self._apply_rat_heuristics(camis, payload)
        return payload
