from fastapi import APIRouter, HTTPException
from config.database import time_series
from models.time_series_result import TimeSeriesResult
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from typing import List


trained_data = APIRouter()


# Retrieves existing data, trains and returns predictive values
@trained_data.get("/GetTrainedWorkLogs", response_model=List[TimeSeriesResult])
async def get_trained_logs():
    output_cursor = time_series.find()
    output = list(output_cursor)

    for item in output:
        item["id"] = str(item["_id"])
    return output
