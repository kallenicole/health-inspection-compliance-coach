# api/models.py
from typing import List, Optional

try:
    # Pydantic v2 style
    from pydantic import BaseModel, ConfigDict
    _HAS_CONFIGDICT = True
except Exception:
    from pydantic import BaseModel  # type: ignore
    _HAS_CONFIGDICT = False


class ViolationProb(BaseModel):
    code: str
    probability: float
    label: str


class ScoreRequest(BaseModel):
    camis: str


class ScoreResponse(BaseModel):
    camis: str
    prob_bc: float
    predicted_points: Optional[float] = None
    top_reasons: List[str] = []
    top_violation_probs: List[ViolationProb] = []

    model_version: Optional[str] = None
    data_version: Optional[str] = None

    last_inspection_date: Optional[str] = None
    last_points: Optional[float] = None
    last_grade: Optional[str] = None

    # NEW rat features
    rat_index: Optional[float] = None
    rat311_cnt_180d_k1: Optional[int] = None
    ratinsp_fail_365d_k1: Optional[int] = None

    # Silence "model_" protected namespace warning in Pydantic
    if _HAS_CONFIGDICT:
        model_config = ConfigDict(protected_namespaces=())
    else:  # v1 fallback
        class Config:  # type: ignore
            protected_namespaces = ()


