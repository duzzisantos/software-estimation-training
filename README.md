# About

This is a backend application that helps to address one of the classical problems in Information Systems: Software Project Estimation Bias. It seeks to collect data from a client application which logs in individual work logs for sub tasks that make up a bigger task.

As task/work time logs are recorded, they are passed through Machine Learning tools like TensorFlow and Scikit-Learn which are notably applied here to perform time-series forecasts using historical work/task time logs.

These trained data are stored as batch data, to enable client view trained data from the past - with the view of detecting anomalies or deviations in the prediction as well as investigating the causes (which might be outside of the application data's scope. Think - what if there was a layoff, or staff were stationed to other projects - thus prolonging the time required to deliver well-known tasks?).

To build this data training model, the Long-Term Short-Term Memory recurrent neural network (RNN) is applied, that way we can predict long-term non-linear series.

## Tools used

- Python
- FastAPI
- MongoDB
- TensorFlow
- Scikit-Learn
- Pandas
- Numpy
- Mathplotlib

## Objectives

- To generate experimental data for a research paper addressing the issue of software project time estimation bias.
- To compare previous academic research outcomes on this topic -
  using results generated from this system.
- To have a system that guides developers and project managers toward making more-informed, objective decisions with regards to estimating effort and time expended in building software.

## Process flow

See flow chart here:
https://ibb.co/XpNCg7Q
