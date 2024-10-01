from pydantic import BaseModel


class PERT(BaseModel):
    optimistic: float | int
    most_likely: float | int
    pessimistic: float | int


class PERTResult(BaseModel):
    simulated_operations: list
    mean_duration: float
    st_deviation: float
    ninetieth_percentile: float
