from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import os
from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv("MONGO_URL")

mongo_client = MongoClient(
    connection_string,
    server_api=ServerApi("1"),
    tlsCAFile=certifi.where(),
)

db = mongo_client.training_result

# Collections under this this database
time_series = db["time-series"]
regression = db["multiple-regression"]
fetch_db = mongo_client.get_database("software_estimation_bias")
work_logs = fetch_db.__getitem__("software_work_log")


try:
    mongo_client.admin.command("ping")
    print("Successfully connected to database")
except Exception as e:
    print(e)
