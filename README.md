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
for Vite are configured in `config.yaml`.

## Run the dashboard

Start the frontend development server:

```powershell
cd frontend
npm run dev
```

The dashboard expects the FastAPI service at `http://localhost:8000`. Set
`VITE_API_BASE_URL` in the frontend environment if the API runs at another URL.

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
