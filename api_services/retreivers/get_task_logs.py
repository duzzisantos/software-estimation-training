from fastapi import APIRouter, HTTPException
from config.database import work_logs
from models.software_tasks import SoftwareTasks
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from typing import List
from etl.middleware import format_for_timeseries
import asyncio

task_router = APIRouter()

## Encapsulate these processes in a task scheduler


# Retrieves instances of task/work logs
@task_router.get("/GetWorkLogsForTraining", response_model=List[SoftwareTasks])
def get_work_logs():
    try:
        work_logs_cursor = work_logs.find()
        logs = list(work_logs_cursor)

        for log in logs:
            log["id"] = str(log["_id"])

        return logs
    except Exception:
        raise ({"Message": "Error occured in processing data"})


formatted_data_for_loading = format_for_timeseries(get_work_logs())


##Start training data here


##After training using time series, submit result to database using another endpoint
