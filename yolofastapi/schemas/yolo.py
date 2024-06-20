from pydantic import BaseModel
from typing import List

class ImageAnalysisResponse(BaseModel):
    id: int
    labels: List[str]
    confidences: List[float]

class FilteredImageAnalysisResponse(BaseModel):
    id: int
    filtered_labels: List[str]
    filtered_confidences: List[float]
    recommendation: str
    percentage: float
