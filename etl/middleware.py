from collections import defaultdict

# from api_services.retreivers.get_task_logs import get_work_logs


## ETL middleware that formats the data which we want to load unto the multivariate time series machine learning model
def format_for_timeseries(data):

    aggregated_data = defaultdict(list)

    for entry in data:
        for key, value in entry.items():
            aggregated_data[key].append(value)

    aggregated_data = dict(aggregated_data)

    aggregated_data["last_updated"] = aggregated_data.pop("last_updated")
    return aggregated_data


## ETL middleware that formats the data which we want to load unto the multi-linear regression machine learning model
