from fastapi import FastAPI
from api_services.retreivers.get_trained_data import task_router
import uvicorn

app = FastAPI()
app.include_router(task_router)
