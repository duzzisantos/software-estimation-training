from fastapi import APIRouter, HTTPException
from config.database import work_logs
from models.software_tasks import SoftwareTasks
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


##Start training data here
data_base_task = formatted_data["database_task"]
security_tasks = formatted_data["security_task"]
validation_tasks = formatted_data["validation_task"]
dev_ops_tasks = formatted_data["dev_ops_task"]
time_period = formatted_data["last_updated"]

data_in_question = {
    "database_task": data_base_task,
    "security_task": security_tasks,
    "validation_task": validation_tasks,
    "devops_task": dev_ops_tasks,
    "last_updated": time_period,
}

##Convert time period to date time
time_converted = pd.to_datetime(data_in_question["last_updated"])

# Create a DataFrame for easier manipulation
df = pd.DataFrame(
    {
        "database_task": data_in_question["database_task"],
        "security_task": data_in_question["security_task"],
        "validation_task": data_in_question["validation_task"],
        "devops_task": data_in_question["devops_task"],
    },
    index=time_converted,
)


## To scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


## Preparing the data for training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


SEQ_LENGTH = 1  # You can adjust this
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Reshape for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
y = y.reshape((y.shape[0], y.shape[1]))

# Build an LSTM model using tensor flow
model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(
            64, input_shape=(SEQ_LENGTH, X.shape[2]), return_sequences=True
        ),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(
            4
        ),  # Four tasks (multivariate)  ## In situation where this done dynamically or inclusive of all data sets, we include all tasks
    ]
)

model.compile(optimizer="adam", loss="mse")

## Train model
history = model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2)

# Predict on the last known values (for the next step)
predicted = model.predict(X[-1].reshape(1, SEQ_LENGTH, X.shape[2]))

# Inverse transform the predicted data back to original scale
predicted_inverse = scaler.inverse_transform(predicted)

# Print the forecasted values for each task - this part goes into the batch inference which is stored in database
print(f"Forecasted database_task: {predicted_inverse[0][0]}")
print(f"Forecasted security_task: {predicted_inverse[0][1]}")
print(f"Forecasted validation_task: {predicted_inverse[0][2]}")
print(f"Forecasted devops_task: {predicted_inverse[0][3]}")


# Visualize the actual vs forecasted values
plt.figure(figsize=(10, 6))

# Plot actual data for the last few time points
plt.plot(
    df.index[-SEQ_LENGTH:],
    df["database_task"].values[-SEQ_LENGTH:],
    label="Actual database_task",
)
plt.plot(
    df.index[-SEQ_LENGTH:],
    df["security_task"].values[-SEQ_LENGTH:],
    label="Actual security_task",
)
plt.plot(
    df.index[-SEQ_LENGTH:],
    df["validation_task"].values[-SEQ_LENGTH:],
    label="Actual validation_task",
)
plt.plot(
    df.index[-SEQ_LENGTH:],
    df["devops_task"].values[-SEQ_LENGTH:],
    label="Actual devops_task",
)

# Forecast (next point based on prediction)
future_time = pd.date_range(df.index[-1], periods=2, freq="W")[
    1
]  # Predict 1 week ahead
plt.scatter(
    future_time, predicted_inverse[0][0], label="Forecast database_task", color="red"
)
plt.scatter(
    future_time, predicted_inverse[0][1], label="Forecast security_task", color="green"
)
plt.scatter(
    future_time, predicted_inverse[0][2], label="Forecast validation_task", color="blue"
)
plt.scatter(
    future_time, predicted_inverse[0][3], label="Forecast devops_task", color="purple"
)

plt.xlabel("Time Period")
plt.ylabel("Task Duration")
plt.title("Multivariate Task Duration Forecast")
plt.legend()
plt.grid(True)
plt.show()


##After training using time series, submit result to database using another endpoint
