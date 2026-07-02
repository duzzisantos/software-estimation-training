from fastapi import APIRouter, Depends
from config.database import work_logs, regression
from middleware.auth import require_api_key, require_unlock_key
from models.regression_result import RegressionResult, InsertResult
from datetime import datetime
from utils import task_labels
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import List


regression_router = APIRouter(dependencies=[Depends(require_api_key), Depends(require_unlock_key)])


def _fetch_work_logs():
    cursor = work_logs.find()
    logs = list(cursor)
    for log in logs:
        log["id"] = str(log["_id"])
    return logs


@regression_router.get("/StoreRegressionResults", response_model=InsertResult)
async def store_regression_results():
    logs = _fetch_work_logs()

    rows = []
    for log in logs:
        row = {}
        for label in task_labels:
            row[label] = log.get(label, 0) or 0
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) < 3:
        return {"id": "insufficient_data"}

    X = df.values
    y = df.sum(axis=1).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    predicted = model.predict(X_scaled)
    r_squared = model.score(X_scaled, y)
    residuals = (y - predicted).tolist()

    result = RegressionResult(
        task_categories=task_labels,
        coefficients=[round(c, 4) for c in model.coef_.tolist()],
        intercept=round(float(model.intercept_), 4),
        r_squared=round(float(r_squared), 4),
        predicted_totals=[round(p, 2) for p in predicted.tolist()],
        actual_totals=[round(a, 2) for a in y.tolist()],
        residuals=[round(r, 2) for r in residuals],
        sample_count=len(df),
        training_date=datetime.now(),
    )

    inserted = regression.insert_one(dict(result))
    return {"id": str(inserted.inserted_id)}


@regression_router.get("/GetRegressionResults", response_model=List[RegressionResult])
async def get_regression_results():
    cursor = regression.find()
    output = list(cursor)
    for item in output:
        item["id"] = str(item["_id"])
    return output
