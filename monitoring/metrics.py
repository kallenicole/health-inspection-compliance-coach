from fastapi import APIRouter
from time import time
router = APIRouter()
START = time()
@router.get("/metrics")
def metrics():
    return {"uptime_seconds": int(time() - START)}
