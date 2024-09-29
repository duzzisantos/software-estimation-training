from pydantic import BaseModel
from datetime import datetime
from typing import List


class ResultItems(BaseModel):
    period: datetime = datetime.isoformat
    worst_case: list[int]
    predicted: list[int]
    best_case: list[int]


class TimeSeriesResult(BaseModel):
    category: str
    latest_results: List[dict]
