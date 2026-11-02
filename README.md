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

## Deploy the backend to Hugging Face Spaces

The root `Dockerfile` is configured for a Docker Space using Python 3.11 and
listening on port `7860`. Deploy the repository to a new Hugging Face Space
with the **Docker** SDK. The Space repository needs the root `Dockerfile`,
`requirements.txt`, `config.yaml`, and `backend/` directory.

The dataset is not tracked by Git. Make
`data/raw/cicids2017_cleaned.csv` available to the Space before starting an
experiment. For example, upload it to the Space repository at that path or
configure persistent Space storage and adjust the dataset path in
`config.yaml`.

The API has no authentication. In the Space settings, set the variable
`IDS_CORS_ORIGINS` to the exact HTTPS origin of the deployed Vercel dashboard,
with no trailing slash. This allows the browser dashboard to call the API from
its separate origin. The setting is optional for local same-origin access.

The API writes settings and results to local files and runs training in an
in-process background thread. Space restarts can reset in-memory run status;
enable persistent storage if experiment outputs should survive restarts. Keep
one API process, and do not expose sensitive data or a public production
workload without adding authentication, rate limits, and run quotas.

## Deploy the frontend to Vercel

The root `vercel.json` builds and serves only the Vite frontend from
`frontend/`; the Python backend is deployed separately to Hugging Face Spaces.
Import the repository into Vercel with the repository root as the project
root. Set the Vercel environment variable `VITE_API_BASE_URL` to the Space's
base URL, for example `https://your-space-name.hf.space` (do not append
`/api`). Set it for the environments you deploy and redeploy after changing
the value. The frontend calls the API's existing `/api/...` endpoints on that
origin. Local frontend development continues to use the API on port `8000`
unless `VITE_API_BASE_URL` is set.

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
