from fastapi import APIRouter
from config.database import work_logs, time_series
from models.software_tasks import SoftwareTasks
from models.time_series_result import TimeSeriesResult
from datetime import datetime
from utils import task_labels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from typing import List
from etl.middleware import format_for_timeseries


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


formatted_data = format_for_timeseries(get_work_logs())


##Start training data here - the list must be exhaustive of all categories
# Fetching tasks from formatted_data
data_base_task = formatted_data["database_task"]
security_tasks = formatted_data["security_task"]
validation_tasks = formatted_data["validation_task"]
dev_ops_tasks = formatted_data["dev_ops_task"]
time_period = formatted_data["last_updated"]
server_management_tasks = formatted_data["server_management_task"]
api_setup_task = formatted_data["api_setup_task"]
api_integration_task = formatted_data["api_integration_task"]
data_backup_task = formatted_data["data_backup_task"]
backend_testing_task = formatted_data["backend_testing_task"]
data_infrastructure_task = formatted_data["data_intucture_task"]
machine_learning_task = formatted_data["machine_learning_task"]
scalability_task = formatted_data["scalability_task"]
optimization_task = formatted_data["optimization_task"]
cloud_task = formatted_data["cloud_task"]

# Frontend tasks
styling_task = formatted_data["styling_task"]
ui_ux_task = formatted_data["ui_ux_task"]
frontend_testing_task = formatted_data["frontend_testing_task"]
api_logic_task = formatted_data["api_logic_task"]
form_setup_task = formatted_data["form_setup_task"]
table_setup_task = formatted_data["table_setup_task"]
layout_setup_task = formatted_data["layout_setup_task"]
data_display_task = formatted_data["data_display_task"]
data_visualization_task = formatted_data["data_visualization_task"]
access_control_task = formatted_data["access_control_task"]
seo_task = formatted_data["seo_task"]
widget_setup_task = formatted_data["widget_setup_task"]
ci_cd_task = formatted_data["ci_cd_task"]
deployment_task = formatted_data["deployment_task"]
cms_integration_task = formatted_data["cms_integration_task"]

# Collecting all tasks into a dictionary
data_in_question = {
    "database_task": data_base_task,
    "security_task": security_tasks,
    "validation_task": validation_tasks,
    "devops_task": dev_ops_tasks,
    "server_management_task": server_management_tasks,
    "api_setup_task": api_setup_task,
    "api_integration_task": api_integration_task,
    "data_backup_task": data_backup_task,
    "backend_testing_task": backend_testing_task,
    "data_infrastructure_task": data_infrastructure_task,
    "machine_learning_task": machine_learning_task,
    "scalability_task": scalability_task,
    "optimization_task": optimization_task,
    "cloud_task": cloud_task,
    "styling_task": styling_task,
    "ui_ux_task": ui_ux_task,
    "frontend_testing_task": frontend_testing_task,
    "api_logic_task": api_logic_task,
    "form_setup_task": form_setup_task,
    "table_setup_task": table_setup_task,
    "layout_setup_task": layout_setup_task,
    "data_display_task": data_display_task,
    "data_visualization_task": data_visualization_task,
    "access_control_task": access_control_task,
    "seo_task": seo_task,
    "widget_setup_task": widget_setup_task,
    "ci_cd_task": ci_cd_task,
    "deployment_task": deployment_task,
    "cms_integration_task": cms_integration_task,
    "last_updated": time_period,
}

# ##Convert time period to date time
time_converted = pd.to_datetime(data_in_question["last_updated"])

# # Create a DataFrame for easier manipulation
# Creating DataFrame from data_in_question with all tasks
df = pd.DataFrame(
    {
        "database_task": data_in_question["database_task"],
        "security_task": data_in_question["security_task"],
        "validation_task": data_in_question["validation_task"],
        "devops_task": data_in_question["devops_task"],
        "server_management_task": data_in_question["server_management_task"],
        "api_setup_task": data_in_question["api_setup_task"],
        "api_integration_task": data_in_question["api_integration_task"],
        "data_backup_task": data_in_question["data_backup_task"],
        "backend_testing_task": data_in_question["backend_testing_task"],
        "data_infrastructure_task": data_in_question["data_infrastructure_task"],
        "machine_learning_task": data_in_question["machine_learning_task"],
        "scalability_task": data_in_question["scalability_task"],
        "optimization_task": data_in_question["optimization_task"],
        "cloud_task": data_in_question["cloud_task"],
        "styling_task": data_in_question["styling_task"],
        "ui_ux_task": data_in_question["ui_ux_task"],
        "frontend_testing_task": data_in_question["frontend_testing_task"],
        "api_logic_task": data_in_question["api_logic_task"],
        "form_setup_task": data_in_question["form_setup_task"],
        "table_setup_task": data_in_question["table_setup_task"],
        "layout_setup_task": data_in_question["layout_setup_task"],
        "data_display_task": data_in_question["data_display_task"],
        "data_visualization_task": data_in_question["data_visualization_task"],
        "access_control_task": data_in_question["access_control_task"],
        "seo_task": data_in_question["seo_task"],
        "widget_setup_task": data_in_question["widget_setup_task"],
        "ci_cd_task": data_in_question["ci_cd_task"],
        "deployment_task": data_in_question["deployment_task"],
        "cms_integration_task": data_in_question["cms_integration_task"],
    },
    index=time_converted,  # Assuming time_converted is defined
)


# ## To scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


# ## Preparing the data for training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


SEQ_LENGTH = 1  # You can adjust this although if the size is too high - then the forecast is infeasible
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Reshape for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
y = y.reshape((y.shape[0], y.shape[1]))

# # Build an LSTM model using tensor flow
model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(
            64, input_shape=(SEQ_LENGTH, X.shape[2]), return_sequences=True
        ),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(
            29
        ),  # Four tasks (multivariate)  ## In situation where this done dynamically or inclusive of all data sets, we include all tasks
    ]
)

model.compile(optimizer="adam", loss="mse")

# ## Train model
history = model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2)

# # Predict on the last known values (for the next step)
predicted = model.predict(X[-1].reshape(1, SEQ_LENGTH, X.shape[2]))

# # Inverse transform the predicted data back to original scale
predicted_inverse = scaler.inverse_transform(predicted)


## Stores training result in database using database schema
@task_router.get("/StoreTrainedResults", response_model=TimeSeriesResult)
async def store_trained_results():
    return TimeSeriesResult(
        task_categories=task_labels,
        predicted_durations=predicted_inverse[0],
        training_date=datetime.now(),
    )
