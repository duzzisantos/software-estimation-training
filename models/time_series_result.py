from pydantic import BaseModel
from datetime import datetime
from typing import List, Union


class InsertResult(BaseModel):
    id: str


class TimeSeriesResult(BaseModel):
    task_categories: List[str]
    predicted_durations: List[Union[float | int]]
    training_date: datetime = datetime.now()
