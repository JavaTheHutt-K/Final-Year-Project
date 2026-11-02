---
title: Adaptive IDS API
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Adaptive Intrusion Detection Framework

This project compares federated and centralised machine-learning models on a lightweight combined CICIDS2017 dataset, with and without a training update. Since the lightweight file has no capture-day metadata, it uses a reproducible stratified random split instead of a Monday–Friday comparison. All experiment settings are defined in the root `config.yaml`.

## Setup

1. Use Python 3.11 and install backend dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Place `cicids2017_cleaned.csv` in `data/raw/`. The configured label field is `Attack Type`; `Normal Traffic` is treated as benign and all other labels as malicious. Dataset files are not tracked by Git.
3. Install the frontend dependencies:
   ```powershell
   cd frontend
   npm install
   ```

Generated cleaning reports and experiment results are written under `results/` and are not tracked by Git.

## Run the backend

From the project root, activate the Python environment and run the complete
configured experiment:

```powershell
python -m backend.run_experiment
```

The CLI cleans and samples the configured CSV in chunks, then makes
reproducible stratified partitions using the configured fractions (default:
70% initial training, 10% update, 20% random holdout). The scaler and constant
feature detection are fit using initial training rows only. It trains the
centralised baseline and Flower FedAvg model, updates both using the update
partition, and evaluates all four versions on the same held-out test set.
Because the file has no capture-day field, the update partition is not newer
traffic and results must not be interpreted as a temporal adaptation test.
Results are saved in `results/results.json`; cleaning and partition counts are
written to `results/data_report.json`; fitted model/scaler artifacts are saved
under `results/models/`.

The default `dataset.sample_fraction` is `0.1` to keep local runs manageable.
Set it to `1.0` in `config.yaml` to use every cleaned row for the full experiment.

The federated experiment simulates the configured clients in one process using
Flower's Ray simulation and FedAvg weighted by client sample count. The configured
logistic-regression model uses scikit-learn's incremental SGD implementation
with logistic loss so model parameters can be transferred and averaged. This is
a simulation, not deployment to real network devices.

Round-level federated charts report metrics averaged across the simulated
clients' training partitions; final comparison metrics are evaluated only on
the random holdout partition.

Start the API from the project root (host and port come from `config.yaml`):

```powershell
python -m backend.api
```

The API reads experiment settings from `config.yaml`. `POST /api/run` starts a
background run; `GET /api/run/status` reports the stage; result and round-history
routes serve completed or in-progress data to the dashboard. The CORS origins
for Vite default to `config.yaml`; set `IDS_CORS_ORIGINS` to a comma-separated
list of exact origins only when the browser accesses the API from another
origin.

## Deploy frontend and API together on Vercel

The root `vercel.json` configures two services in one Vercel project:

- `frontend` builds the Vite application from `frontend/` and serves public
  paths not starting with `/api/`.
- `app` runs the FastAPI application from `backend.api:app`; public `/api/*`
  requests are routed to it.

Import the repository into Vercel with the repository root as the project root.
The Vercel frontend uses same-origin API paths in production, so no
`VITE_API_BASE_URL` setting is required. Local frontend development continues
to use the API on port `8000`; set `VITE_API_BASE_URL` only when intentionally
connecting the browser to an API on another origin.

The current API writes settings and experiment results to local files and
starts training in an in-process background thread. Vercel Functions do not
provide durable local storage or a durable background worker, so these
operations are not reliable in this deployment as currently implemented.
Use external persistent storage and a durable job runner (or deploy the
stateful API on a host that supports them) before relying on settings updates
or experiment runs in production. The API also requires the Flower/Ray
simulation dependencies and dataset; check Vercel's Python function and build
limits before choosing to host the API service there.

This API currently has **no authentication**. Anyone who can reach its public
routes can read available data, change experiment settings, and start
resource-intensive runs. Do not expose it to sensitive data or a public
production workload without adding access control, rate limits, and run quotas.
Generated experiment status is stored in memory and is reset whenever the
Space process restarts; use one API process.

## Run the dashboard

Start the frontend development server:

```powershell
cd frontend
npm run dev
```

By default, local development contacts the API on port `8000` at the frontend's
hostname. Set `VITE_API_BASE_URL` at build time when the API is hosted elsewhere.

## Frontend production build

```powershell
cd frontend
npm run build
npm run preview
```

## Run backend tests

```powershell
python -m pytest
```
