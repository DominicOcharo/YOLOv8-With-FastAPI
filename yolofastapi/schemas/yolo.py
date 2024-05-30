# schemas/yolo.py
from pydantic import BaseModel
from typing import List 

class ImageAnalysisResponse(BaseModel):
    id: int
    labels: List[str]
