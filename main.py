from fastapi import FastAPI
from api_services.retreivers.get_trained_data import task_router
from api_services.retreivers.get_pert import pert_router
import uvicorn

app = FastAPI()
app.include_router(task_router)
app.include_router(pert_router)
