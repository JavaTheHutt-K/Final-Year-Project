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
list of exact origins when deploying.

## Hosting on Hugging Face Spaces and Vercel

The backend is configured as a Docker Space listening on port `7860`; the
frontend is a static Vite application.

### Deploy the backend to Hugging Face Spaces

1. Push this repository to GitHub, then create a Hugging Face Space with the
   **Docker** SDK and connect it to the repository (or push the repository to
   the Space). The root `Dockerfile` and this file's Space metadata configure
   the service.
2. In Space **Settings**, choose hardware with enough RAM and disk for the
   dataset and Flower/Ray simulation. The simulation may exceed free hardware
   limits; check build logs and runtime logs if the Space fails to build or run.
3. Make the dataset available at
   `data/raw/cicids2017_cleaned.csv`. Dataset files are git-ignored; upload the
   file to the Space/repository separately, or use persistent Space storage
   and update the data paths in `config.yaml` to match its mount location.
4. Once the Vercel site is deployed, set the Space variable `IDS_CORS_ORIGINS`
   to its exact HTTPS origin, for example `https://your-project.vercel.app`.
   Multiple origins can be comma-separated. Do not include a trailing slash.
5. Wait for the Space to rebuild. Check its `/health` endpoint; it should return
   `{"status":"ok"}`.

Spaces without persistent storage can lose generated result and model files
when the container is rebuilt or restarted. Enable persistent storage if
available and configure `paths.results_dir`, `paths.data_report`, and
`paths.results_file` in `config.yaml` to point inside its mount.

### Deploy the frontend to Vercel

1. Import the GitHub repository into Vercel and set **Root Directory** to
   `frontend`.
2. Use build command `npm run build` and output directory `dist`. Vercel
   detects Vite automatically; `frontend/vercel.json` sends app routes to the
   SPA entry point.
3. Add the environment variable `VITE_API_BASE_URL` with the full Hugging Face
   Space URL, for example `https://your-space-name.hf.space` (do not append
   `/api`).
4. Deploy or redeploy after setting the variable. Open the Vercel site and
   verify dashboard data loads from the Space.

This API currently has **no authentication**. Anyone who can reach the Space
can read its data, change experiment settings, and start resource-intensive
runs. Do not use a public Space with sensitive data or expose it for a public
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
