from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from typing import List, Optional

class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class ScoreRequest(APIModel):
    camis: str = Field(..., description="NYC CAMIS restaurant identifier")

class ViolationProb(APIModel):
    code: str
    probability: float
    label: str

class ScoreResponse(APIModel):
    camis: str
    prob_bc: float
    predicted_points: Optional[float] = None
    top_reasons: List[str]
    top_violation_probs: List[ViolationProb]
    model_version: str
    data_version: str
    last_inspection_date: Optional[str] = None
    last_points: Optional[int] = None
    last_grade: Optional[str] = None
