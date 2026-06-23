from pydantic import BaseModel
from datetime import datetime
from typing import List, Union


class InsertResult(BaseModel):
    id: str


class RegressionResult(BaseModel):
    task_categories: List[str]
    coefficients: List[float]
    intercept: float
    r_squared: float
    predicted_totals: List[float]
    actual_totals: List[float]
    residuals: List[float]
    sample_count: int
    training_date: datetime = datetime.now()
