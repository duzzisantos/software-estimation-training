# Software Project Estimator — Training Service

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Machine learning service that tackles software project estimation bias through PERT analysis, LSTM time-series forecasting, and multilinear regression.**

[Research Paper](https://drive.google.com/file/d/1bNhzLDKLmjnZvtZxHhtVjy19-8iR8DUO/view?usp=drive_link)

</div>

---

## The Problem

Software project estimation is notoriously inaccurate. Teams consistently underestimate effort and timelines — a well-documented phenomenon in information systems research known as **estimation bias**. This service applies statistical and machine learning techniques to historical work logs, producing data-driven forecasts that help developers and project managers make more objective decisions.

---

## Architecture

```
                    ┌───────────────────────────────────────┐
                    │           Client Application          │
                    │     React Dashboard  ·  Swagger UI    │
                    └────────────────┬──────────────────────┘
                                     │
                                  API Calls
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
┌────────▼────────┐       ┌──────────▼──────────┐     ┌──────────▼──────────┐
│  PERT Analysis  │       │  Time Series        │     │  Multiple Linear    │
│                 │       │  Forecasting         │     │  Regression         │
│  Monte Carlo    │       │                     │     │                     │
│  Simulation     │       │  LSTM Neural        │     │  Feature            │
│  (10,000 runs)  │       │  Network (RNN)      │     │  Coefficient        │
│                 │       │                     │     │  Analysis           │
│  Triangular     │       │  2-Layer Deep       │     │                     │
│  Distribution   │       │  Network            │     │  R² Scoring &       │
│                 │       │  (64 → 32 units)    │     │  Residual Analysis  │
└────────┬────────┘       └──────────┬──────────┘     └──────────┬──────────┘
         │                           │                           │
         └───────────────┬───────────┴───────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │      ETL Layer      │
              │   Data formatting   │
              │   Feature scaling   │
              │   Sequence prep     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │      MongoDB        │
              │                     │
              │  ┌───────────────┐  │
              │  │ work_logs     │  │
              │  │ (source data) │  │
              │  ├───────────────┤  │
              │  │ time-series   │  │
              │  │ (predictions) │  │
              │  ├───────────────┤  │
              │  │ regression    │  │
              │  │ (model output)│  │
              │  └───────────────┘  │
              └─────────────────────┘
```

---

## Models

### PERT Analysis

Applies the **Program Evaluation and Review Technique** with Monte Carlo simulation to estimate task durations under uncertainty.

```
Input                          Process                         Output
─────                          ───────                         ──────
Optimistic time (a)     ┐
Most likely time (m)    ├───▶  Triangular distribution  ───▶  Mean duration
Pessimistic time (b)    ┘      × 10,000 simulations           Std deviation
                                                               90th percentile
```

- Accepts three time estimates per task (optimistic, most likely, pessimistic)
- Generates 10,000 random samples using triangular distribution
- Returns statistical summary: mean, standard deviation, P90

### LSTM Time-Series Forecasting

Predicts future task durations using a **Long Short-Term Memory** recurrent neural network trained on historical work logs.

```
Input                          Architecture                    Output
─────                          ────────────                    ──────
Historical work logs    ───▶   LSTM Layer 1 (64 units)  ───▶  Predicted durations
(29 task categories)           LSTM Layer 2 (32 units)         per task category
                               Dense Layer (29 units)          for next period
```

- Processes 29 task categories across backend and frontend domains
- Two-layer deep LSTM captures long-term non-linear patterns
- Batch predictions are stored historically, enabling anomaly detection across training periods

### Multiple Linear Regression

Identifies which task categories most influence total project duration.

```
Input                          Analysis                        Output
─────                          ────────                        ──────
29 task features        ───▶   Feature coefficients     ───▶  R² score
(time per category)            Predicted vs Actual             Residual analysis
                               Residual calculation            Top influencers
```

- Each of the 29 task types serves as an independent variable
- Calculates per-feature coefficients to rank task impact
- Produces R² score, intercept, and residual distribution

---

## Task Categories

The model tracks **29 standardized subtask types** across two domains:

| Backend | Frontend |
|---------|----------|
| Database | Styling |
| Security | UI/UX |
| Validation | Frontend Testing |
| DevOps | API Logic |
| Server Management | Form Setup |
| API Setup | Table Setup |
| API Integration | Layout Setup |
| Data Backup | Data Display |
| Backend Testing | Data Visualization |
| Data Structure | Access Control |
| Machine Learning | SEO |
| Scalability | Widget Setup |
| Optimization | CI/CD |
| Cloud | Deployment |
| | CMS Integration |

---

## API Reference

### PERT

<details>
<summary><strong>POST /PertAnalysis</strong></summary>

Run a PERT analysis with Monte Carlo simulation.

**Request:**
```json
{
  "selected_tasks": ["database_task", "api_setup_task"],
  "optimistic": 5,
  "most_likely": 10,
  "pessimistic": 20
}
```

**Response:**
```json
{
  "mean_duration": 11.67,
  "std_deviation": 3.12,
  "percentile_90": 15.83,
  "simulations": [10.2, 11.5, 9.8, ...]
}
```

</details>

### Time Series

<details>
<summary><strong>GET /GetTrainedWorkLogs</strong></summary>

Retrieve stored LSTM predictions from previous training runs.

</details>

<details>
<summary><strong>GET /StoreTrainedResults</strong></summary>

Trigger LSTM training on current work logs and store the predictions.

</details>

<details>
<summary><strong>POST /Retrain</strong></summary>

Trigger a background retraining of the LSTM model.

</details>

### Regression

<details>
<summary><strong>GET /StoreRegressionResults</strong></summary>

Run multiple linear regression and store coefficients, R², and residuals.

</details>

<details>
<summary><strong>GET /GetRegressionResults</strong></summary>

Retrieve stored regression analysis results.

</details>

### Data

<details>
<summary><strong>GET /GetWorkLogsForTraining</strong></summary>

Retrieve raw work logs formatted for model training.

</details>

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd software-estimation-training
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
# Set MONGODB_URI and other variables in .env

# 3. Run
python main.py
# → http://localhost:3000/docs
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGODB_URI` | Yes | MongoDB connection string |
| `DB_NAME` | Yes | Database name for work logs |
| `TRAINING_DB_NAME` | Yes | Database name for model outputs |

---

## Research Context

This system generates experimental data for research into software project estimation bias. The objective is to compare outcomes against established academic findings and provide a practical tool for data-driven project planning.

See the full research paper: [Software Project Estimation Bias](https://drive.google.com/file/d/1bNhzLDKLmjnZvtZxHhtVjy19-8iR8DUO/view?usp=drive_link)

---

## License

[MIT](LICENSE)
