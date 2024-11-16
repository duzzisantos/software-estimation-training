from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from api_services.retreivers.get_trained_data import task_router
from api_services.retreivers.get_pert import pert_router
from api_services.retreivers.get_time_series import trained_data
import uvicorn
import os
import signal
import time
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
website_url = os.getenv("WEBSITE_URL")

origins = [
    "http://localhost:5173",
    website_url,
    ##and other origins: eg production, staging etc
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def touch_file():
    with open("reload.txt", "a") as f:
        f.write(f"Refreshed at {time.time()}\n")


@app.post("/Retrain")
async def refresh_server(background_tasks: BackgroundTasks):
    background_tasks.add_task(touch_file)
    return {"message": "Server is refreshing..."}


app.include_router(task_router)
app.include_router(pert_router)
app.include_router(trained_data)
