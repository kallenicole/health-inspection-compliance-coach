from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from typing import List, Optional

class ScoreRequest(BaseModel):
    camis: str = Field(..., description="NYC CAMIS restaurant identifier")

class ViolationProb(BaseModel):
    code: str
    probability: float
    label: str

class ScoreResponse(BaseModel):
    camis: str
    prob_bc: float
    predicted_points: Optional[float] = None
    top_reasons: List[str]
    top_violation_probs: List[ViolationProb]
    model_version: str
    data_version: str
