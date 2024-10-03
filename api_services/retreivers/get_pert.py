import numpy as np
from typing import Union
from models.pert_model import PERT
from fastapi import APIRouter


## Program that estimates time it takes to perform software project time using PERT Analysis and Monte Carlo Simulations
## The random sampling is triangular, and the inference is dynamic
def triangular_distribution(
    optimistic: Union[float | int],
    most_likely: Union[float | int],
    pessimistic: Union[float | int],
):
    return np.random.triangular(optimistic, most_likely, pessimistic)


## This is the number of iteration we are going to make in the probability distribution function.
## It represents how many times a random sampling will be done.
simulation_count = 10000


def monte_carlo_pert(
    optimistic: Union[float | int],
    most_likely: Union[float | int],
    pessimistic: Union[float | int],
    iterations=simulation_count,
):

    ##Run monte carlo simulation for a task by randomly sampling duration for task in each iteration
    results = []
    for _ in range(iterations):
        duration = triangular_distribution(optimistic, most_likely, pessimistic)
        results.append(duration)
    return results


pert_router = APIRouter()


## API for generating PERT simulation in frontend
@pert_router.post("/PertAnalysis")
async def run_pert_analysis(item: PERT):

    try:
        if item.pessimistic != 0 or item.optimistic != 0 or item.most_likely != 0:
            simulated_operations = monte_carlo_pert(
                item.optimistic, item.most_likely, item.pessimistic
            )
            mean_project_duration = np.mean(simulated_operations)
            standard_deviation_duration = np.std(simulated_operations)
            percentile_90 = np.percentile(simulated_operations, 90)

            final_result = {
                "simulated_operations": simulated_operations,
                "predictions": {
                    "mean_duration": mean_project_duration,
                    "st_deviation": standard_deviation_duration,
                    "ninetieth_percentile": percentile_90,
                },
                "pessimistic_estimation": item.pessimistic,
                "most_likely_estimation": item.most_likely,
                "optimistic_estimation": item.optimistic,
            }

            return dict(final_result)
        else:
            return {
                "Message": "Request body cannot be empty, please provide estimates!"
            }

    except Exception as e:
        return e
