---
title: Adaptive IDS API
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: 3.10.13
app_file: app.py
---

# Adaptive Intrusion Detection Framework

This project compares federated and centralised machine-learning models on a lightweight combined CICIDS2017 dataset, with and without a training update. Since the lightweight file has no capture-day metadata, it uses a reproducible stratified random split instead of a Monday–Friday comparison. All experiment settings are defined in the root `config.yaml`.

For Google Cloud Run backend deployment with the React/Vite frontend hosted on
Vercel, see [README_DEPLOYMENT.md](README_DEPLOYMENT.md).

## Setup

1. Use Python 3.11 locally and install project and test dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-dev.txt
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
routes serve completed or in-progress data to the dashboard. Configure the
browser origin with `FRONTEND_URL`, or use `IDS_CORS_ORIGINS` for a comma-separated
list of exact origins. For Cloud Run deployment configuration and limitations, see
[README_DEPLOYMENT.md](README_DEPLOYMENT.md).

The results page can optionally request an AI explanation of a completed
centralised-versus-federated comparison. Configure `AI_API_KEY` on the backend
(and optionally `AI_MODEL` and `AI_BASE_URL`) to enable it. The request sends only
the selected test set's summary metrics to an OpenAI-compatible chat completions
endpoint; it does not send the dataset or model artifacts. AI analysis is
user-triggered, is unavailable during an experiment, and never changes measured
metrics or model training. Without the API key, normal experiment and dashboard
functionality remains available.

## Deploy the frontend to Vercel

The root `vercel.json` builds and serves only the Vite frontend from
`frontend/`; the Python backend is deployed separately to Google Cloud Run.
Import the repository into Vercel with the repository root as the project
root. Set the Vercel environment variable `VITE_API_BASE_URL` to the Cloud Run
service URL (do not append `/api`). Set it for the environments you deploy and
redeploy after changing the value. The frontend calls the API's existing
`/api/...` endpoints on that origin. Local frontend development continues to
use the API on port `8000` unless `VITE_API_BASE_URL` is set.

The dashboard's administrative operations use short-lived bearer tokens issued
by the backend. Keep `ADMIN_PASSWORD_HASH`, `ADMIN_TOKEN_SECRET`,
`SUPABASE_KEY`, and `GEMINI_API_KEY` server-side; they must never be added to
frontend environment variables. Experiment status and results are read from
Supabase when configured, so they remain consistent across API instances.

AI Insights and model comparison use Gemini only when `GEMINI_API_KEY` is
configured on the backend. If the key is missing or Gemini is unavailable due
to quota, rate limits, model availability, network failure, or an invalid
response, the API returns a deterministic interpretation derived from saved
experiment metrics. Responses identify the source as either `gemini` or
`deterministic-fallback`; no paid AI provider is used.

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
