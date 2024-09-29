from pydantic import BaseModel


class TaskModel(BaseModel):
    task_category: str
    description: str
    submitted_by: str
    duration: int
